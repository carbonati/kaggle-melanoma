import os
import numpy as np
import pandas as pd
import cv2
import pickle
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer


def load_data(filepath,
              duplicate_path=None,
              cv_folds_dir=None):
    df_mela = pd.read_csv(filepath)

    # need to add more logic here for the test set
    if duplicate_path is not None:
        df_dupes = pd.read_csv(duplicate_path)
        image_ids_duped = df_dupes['ISIC_id_paired'].tolist()
        df_mela = df_mela.loc[~df_mela['image_name'].isin(image_ids_duped)].reset_index(drop=True)

    if cv_folds_dir is not None:
        cv_folds = load_cv_folds(os.path.join(cv_folds_dir, 'cv_folds.p'))
        df_mela['fold'] = get_fold_col(df_mela, cv_folds)

    return df_mela


def load_image(root, image_id, bgr2rgb=True):
    img = cv2.imread(os.path.join(root, f'{image_id}.jpg'))
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


def generate_cv_folds(df,
                      test_size=0.1,
                      num_folds=10,
                      train_cols=None,
                      stratify_test=None,
                      stratify_val=None,
                      index_col='patient_id',
                      agg_fnc='max',
                      random_state=42069):
    """Generates a list of `num_folds` with `index_col` values stratified by
    the specified columns
    """
    if train_cols is not None:
        if not isinstance(train_cols, (list, tuple)):
            train_cols = [train_cols]
        train_ids_force = set()
        for col in train_cols:
            train_ids_force.update(df.loc[df[col] == 1, index_col].tolist())
        train_ids_force = list(train_ids_force)
    else:
        train_ids_force = []

    df_fold = df.loc[~df[index_col].isin(train_ids_force)].set_index(index_col)

    if test_size > 0:
        if stratify_test is not None:
            if not isinstance(stratify_test, (tuple, list)):
                stratify_test = [stratify_test]

            df_fold = df_fold.groupby(index_col)[stratify_test].agg(agg_fnc)
            targets = df_fold.apply(
                lambda x: '_'.join([str(x[col]) for col in stratify_test]),
                axis=1
            ).values
        else:
            targets = None

        train_idx, test_idx = train_test_split(df_fold.index,
                                               stratify=targets,
                                               test_size=test_size,
                                               random_state=random_state)
    else:
        train_idx = df_fold.index
        test_idx = []

    # add back train ID's to use for CV
    train_idx = list(train_idx) + train_ids_force

    if stratify_val is not None:
        if not isinstance(stratify_val, (tuple, list)):
            stratify_val = [stratify_val]

        df_train = df.loc[df[index_col].isin(train_idx)]
        df_train = df_train.groupby(index_col)[stratify_val].agg(agg_fnc)
        # update target values to stratify on the updated train set
        targets = df_train.apply(
            lambda x: '_'.join([str(x[col]) for col in stratify_val]),
            axis=1
        ).values
        kf = StratifiedKFold(num_folds,
                             random_state=random_state,
                             shuffle=True)
    else:
        df_train = df.loc[df[index_col].isin(train_idx)].set_index(index_col)
        targets = None
        kf = KFold(num_folds,
                   random_state=random_state,
                   shuffle=True)

    cv_folds = []
    for i, (tr_idx, val_idx) in enumerate(kf.split(df_train.index, y=targets)):
        cv_folds.append(df_train.index[val_idx].tolist())
    # add test id's as the last fold
    cv_folds.append(list(test_idx))
    return cv_folds


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
