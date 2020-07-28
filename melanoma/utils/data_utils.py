import os
import json
import numpy as np
import pandas as pd
import cv2
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def load_data(filepath,
              duplicate_path=None,
              cv_folds_dir=None,
              keep_prob=1):
    df_mela = pd.read_csv(filepath)

    # need to add more logic here for the test set
    if duplicate_path is not None:
        df_dupes = pd.read_csv(duplicate_path)
        image_ids_duped = df_dupes['ISIC_id_paired'].tolist()
        df_mela = df_mela.loc[~df_mela['image_name'].isin(image_ids_duped)].reset_index(drop=True)

    # subset for testing
    if keep_prob < 1 and keep_prob > 0:
        df_mela = df_mela.sample(frac=keep_prob, replace=False).reset_index(drop=True)

    if cv_folds_dir is not None:
        cv_folds = load_cv_folds(os.path.join(cv_folds_dir, 'cv_folds.p'))
        df_mela['fold'] = get_fold_col(df_mela, cv_folds)

    df_mela = fill_na(df_mela, 'anatom_site_general_challenge', how='missing')
    return df_mela


def load_image(root, image_id, img_format='jpg', bgr2rgb=False):
    img = cv2.imread(os.path.join(root, f'{image_id}.{img_format}'))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_img_stats(img):
    x = img.reshape(-1, 3).astype(np.float32) / 255.
    meta = {
        'mean': list(x.mean(axis=0)),
        'std': list(x.std(axis=0)),
        'max_pixel_value': x.max()
    }
    return meta


def trim_img(img):
    h, w, d = img.shape
    new_dim = min(h, w)
    h_new = (h - new_dim) // 2
    w_new = (w - new_dim) // 2

    if h_new > 0:
        img = img[h_new:-h_new, :]
    if w_new > 0:
        img = img[:, w_new:-w_new]
    return img


def resize_img(img, size, interpolation=cv2.INTER_AREA, return_meta=True):
    """Resizes and image and returns normalization statistics.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    size : int, tuple/list
        Output image size.
    interpolation : int (default=cv2.INTER_LANCZOS4)
        Interpolation method.
    return_meta : boolean (default=True)
        Boolean whether to return image statistics.

    Returns
    -------
    img : np.ndarray
        Resized output image.
    meta : dict
        Dictionary of the resized images mean and std if `return_meta` is true.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    img = trim_img(img)
    img = cv2.resize(img, size, interpolation=interpolation)

    if return_meta:
        meta = get_img_stats(img)
        return img, meta
    else:
        return img


def load_cv_folds(filepath):
    with open(filepath, 'rb') as f:
        cv_folds = pickle.load(f)
    return cv_folds


def get_fold_col(df, cv_folds, index_col='patient_id'):
    num_folds = len(cv_folds) - 1
    index = df.index if index_col is None else df[index_col]
    s = pd.Series([None] * len(df), index=index)
    for i, val_idx in enumerate(cv_folds):
        if i < num_folds:
            s.loc[s.index.isin(val_idx)] = i
        else:
            s.loc[s.index.isin(val_idx)] = 'test'
    return s.values


def fold_agg_fnc(x):
    agg_dict = {
        'num_images': len(x),
        'num_patients': x['patient_id'].nunique(),
        'pos_rate': x['target'].mean()
    }
    return pd.Series(agg_dict)


def bin_continuous_feature(x, n_bins, strategy='quantile', apply_log=False):
    if apply_log:
        x = np.log(x+1)
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    return est.fit_transform(np.asarray(x).reshape(-1,1))


def patient_agg_fnc(x):
    agg_dict = {
        'num_images': len(x),
        'malignant_count': sum(x['target']),
        'malignant_rate': sum(x['target']) / len(x),
        'age_mean': x['age_approx'].mean(),
        'age_std': x['age_approx'].std()
    }
    return pd.Series(agg_dict)


def fill_na(df, col, how='mean'):
    if how == 'missing':
        df.loc[df[col].isnull(), col] = 'missing'
    else:
        df.loc[df[col].isnull(), col] = df[col].agg(how)
    return df


def get_df_agg(df, colname, value, how='count', index='patient_id'):
    if how == 'count':
        df_agg =  df.groupby([index, colname])[value].size().reset_index()
    elif how =='idxmax':
        df_agg =  df.groupby([index, colname])[value].size().reset_index()
        df_agg = df_agg.loc[df_agg.groupby([index])[value].idxmax()].reset_index(drop=True)
    elif how =='idxmin':
        df_agg =  df.groupby([index, colname])[value].size().reset_index()
        df_agg = df_agg.loc[df_agg.groupby([index])[value].idxmin()].reset_index(drop=True)
    else:
        raise ValueError(f'Unrecognized `how`')

    return df_agg


def stratify_batches(indices,
                     labels,
                     batch_size,
                     drop_last=False,
                     random_state=6969):
    """Returns a list of indices stratified by `labels` where each stratification is
    of size `bathc_size`
    """
    strat_indices = []

    num_batches = int(np.ceil(len(indices) / batch_size))
    remainder = len(indices) % batch_size
    num_batches = num_batches - 1 if remainder > 0 else num_batches

    if remainder > 0:
        remainder_indices = []
        remainder_labels = []
        rs = np.random.RandomState(5)
        last_idx = rs.choice(indices, size=remainder, replace=False)
        for idx in indices:
            if idx not in last_idx:
                remainder_indices.append(idx)
                remainder_labels.append(labels[idx])
    else:
        remainder_indices = indices
        remainder_labels = labels
        last_idx = []

    skf = StratifiedKFold(n_splits=num_batches,
                          shuffle=True,
                          random_state=random_state)

    for _, batch_idx in skf.split(remainder_indices, remainder_labels):
        strat_indices.extend(batch_idx)

    if not drop_last:
        strat_indices.extend(last_idx)

    return strat_indices


def load_img_stats(root, fold_id, filename='img_stats.json'):
    filepath = os.path.join(root, f'fold_{fold_id}', filename)
    print(f'Loading img stats for fold {fold_id} from {filepath}')
    with open(filepath, 'rb') as f:
        img_stats = eval(json.load(f))
    return img_stats
