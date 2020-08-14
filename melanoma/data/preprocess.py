import os
import json
import uuid
import numpy as np
import pandas as pd
import cv2
import torch
from zipfile import ZipFile
from tqdm import tqdm
from skimage.measure import regionprops
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from utils import data_utils, model_utils, train_utils
from utils import generic_utils as utils
from data import dataset as melanoma_dataset
import config as melanoma_config


def preprocess_images(train_image_dir,
                      train_filepath,
                      output_dir,
                      size,
                      interpolation=4,
                      img_format='jpg',
                      quality=100,
                      return_meta=True,
                      cropped_dir=None,
                      cropped_dim=None,
                      test_image_dir=None,
                      test_filepath=None,
                      duplicate_path=None,
                      bgr2rgb=True,
                      keep_prob=1):
    """Resize images and save metadata to disk."""
    compression = cv2.IMWRITE_PNG_COMPRESSION if img_format == 'png' else cv2.IMWRITE_JPEG_QUALITY

    if size is not None:
        base_output_dir = os.path.join(output_dir, f'{size}x{size}_{img_format}_{quality}_{interpolation}')
        train_output_path = os.path.join(base_output_dir, 'train.zip')
        test_output_path = os.path.join(base_output_dir, 'test.zip')
    else:
        base_output_dir = os.path.join(output_dir, f'original_{interpolation}')
        train_output_path = None
        test_output_path = None

    meta_dir = os.path.join(base_output_dir, 'metadata')
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    df_train = data_utils.load_data(train_filepath,
                                    duplicate_path=duplicate_path,
                                    keep_prob=keep_prob)
    df_train['image_dir'] = train_image_dir
    df_train['partition'] = 'train'
    if test_filepath is not None:
        df_test = data_utils.load_data(test_filepath, keep_prob=keep_prob)
        df_test['image_dir'] = test_image_dir
        df_test['partition'] = 'test'
        df_mela = pd.concat(
            (df_train[['image_name', 'image_dir', 'partition']], df_test[['image_name', 'image_dir', 'partition']]),
            axis=0,
            ignore_index=True
        )
    else:
        df_mela = df_train[['image_name', 'image_dir', 'partition']].copy()

    print(f'Saving train images to {train_output_path}')
    print(f'Saving test images to {test_output_path}')
    if return_meta:
        print(f'Saving metadata to {meta_dir}')

    cropped_coords = None
    with ZipFile(train_output_path, 'w') as train_file, ZipFile(test_output_path, 'w') as test_file:
        for row in tqdm(df_mela.itertuples(), total=len(df_mela), desc='Preprocessing images'):
            image_id = row.image_name

            print(image_id)
            img = data_utils.load_image(row.image_dir, image_id, bgr2rgb=bgr2rgb)
            if size is None:
                img = data_utils.trim_img(img)
                meta = data_utils.get_img_stats(img)
            else:
                if cropped_dir is not None:
                    with open(os.path.join(cropped_dir, row.partition, f'{image_id}.json'), 'r') as f:
                        cropped_coords = json.load(f)
                img, meta = data_utils.resize_img(img,
                                                  size=size,
                                                  cropped_coords=cropped_coords,
                                                  cropped_dim=cropped_dim,
                                                  interpolation=interpolation)

                img = cv2.imencode(f'.{img_format}', img, [compression, quality])[1]
                if row.partition == 'train':
                    train_file.writestr(f'{image_id}.{img_format}', img)
                else:
                    test_file.writestr(f'{image_id}.{img_format}', img)

            if return_meta:
                with open(os.path.join(meta_dir, f'{image_id}.json'), 'w') as f:
                    json.dump(str(meta), f)

    print(f'Saved output to {base_output_dir}')


def generate_cv_folds(df,
                      test_size=0.1,
                      num_folds=10,
                      train_map=None,
                      stratify_test=None,
                      stratify_val=None,
                      index_col='patient_id',
                      agg_fnc='max',
                      random_state=42069):
    """Generates a list of `num_folds` with `index_col` values stratified by
    the specified columns

    Parameters
    ----------
    train_map : dict (default=None)
        Dictionary where each key is a column in `df` mapped to a value where
        any sample in that satifies the mapping is forced in to the train set
    """
    if train_map is not None:
        train_ids_force = set()
        for k, v in train_map.items():
            train_ids_force.update(df.loc[df[k] == v, index_col].tolist())
        train_ids_force = list(train_ids_force)
    else:
        train_ids_force = []

    df_fold = df.loc[~df[index_col].isin(train_ids_force)].set_index(index_col)

    if test_size > 0:
        if stratify_test is not None:
            if not isinstance(stratify_test, (tuple, list)):
                stratify_test = [stratify_test]

            # df_fold.to_csv('df_cv.csv')
            df_fold = df_fold.groupby(index_col)[stratify_test].agg(agg_fnc)
            targets = df_fold.apply(
                lambda x: '_'.join([str(x[col]) for col in stratify_test]),
                axis=1
            ).values
        else:
            targets = None

        df_fold['stratify'] = targets
        targets_bad = df_fold['stratify'].value_counts().index[df_fold['stratify'].value_counts() < 2]
        if len(targets_bad) == 1:
            df_fold = df_fold.loc[df_fold['stratify'] != targets_bad[0]]
        df_fold.loc[df_fold['stratify'].isin(targets_bad), 'stratify'] = 'other'
        train_idx, test_idx = train_test_split(df_fold.index,
                                               stratify=df_fold['stratify'],
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


def generate_img_stats(df, target_col='target', name=None):
    img_mean = np.zeros(3, np.float32)
    img_var = np.zeros(3, np.float32)
    num_samples = len(df)

    for row in tqdm(df.itertuples(), total=num_samples, desc=name):
        with open(os.path.join(row.meta_dir, f'{row.image_name}.json'), 'r') as f:
            meta = eval(json.load(f))
        img_mean += meta['mean']
        img_var += np.asarray(meta['std'])**2

    img_mean /= num_samples
    img_var /= num_samples
    img_stats = {
        'n': num_samples,
        'prior': df[target_col].mean().astype(np.float32),
        'mean': list(np.asarray(img_mean)),
        'std': list(np.asarray(np.sqrt(img_var))),
        'max_pixel_value': 1
    }
    return img_stats


def prepare_isic_2018(root, output):
    df_meta = pd.read_csv(os.path.join(root, 'HAM10000_metadata.csv'))
    df_meta['target'] = (df_meta['dx'] == 'mel').astype(int)
    df_meta['source'] = 'ISIC_2018'
    df_meta['localization'] = df_meta['localization'].map(melanoma_config.ANATOM_MAP).fillna(df_meta['localization'])
    df_meta['localization'] = df_meta['localization'].fillna('unknown')
    # assign a new patient_id
    df_meta['patient_id'] = [str(uuid.uuid4()) for _ in range(len(df_meta))]

    df_meta = df_meta.rename(columns={
        'image_id': 'image_name',
        'age': 'age_approx',
        'localization': 'anatom_site_general_challenge'
    })
    filepath = os.path.join(output, 'isic_2018.csv')
    print(f'Saving 2018 ISIC train file to {filepath}')
    df_meta.to_csv(filepath, index=False)


def prepare_isic_2019(root, output):
    df_train = pd.read_csv(os.path.join(root, 'ISIC_2019_Training_GroundTruth.csv'))
    df_meta = pd.read_csv(os.path.join(root, 'ISIC_2019_Training_Metadata.csv'))
    df_train['target'] = df_train['MEL'].astype(int).copy()
    df_train = pd.merge(df_meta, df_train[['image', 'target']], how='left', on='image')
    df_train['source'] = 'ISIC_2019'
    df_train['anatom_site_general'] = df_train['anatom_site_general'].map(melanoma_config.ANATOM_MAP).fillna(df_train['anatom_site_general'])
    df_train['anatom_site_general'] = df_train['anatom_site_general'].fillna('unknown')
    # assign a new patient_id
    df_train['patient_id'] = [str(uuid.uuid4()) for _ in range(len(df_train))]

    df_train = df_train.rename(columns={
        'image': 'image_name',
        'anatom_site_general': 'anatom_site_general_challenge'
    })

    filepath = os.path.join(output, 'isic_2019.csv')
    print(f'Saving 2019 ISIC train file to {filepath}')
    df_train.to_csv(filepath, index=False)


def prepare_v2_malignant(root, output):
    df_train = pd.read_csv(os.path.join(root, 'train_malig_2.csv'))
    df_train['source'] = 'v2_malignant'
    return df_train


def get_crop_coords(weights, img_size=224, pad=None):
    num_tiles = int(np.sqrt(weights.shape[1]))
    upscaled = cv2.resize(weights.reshape(num_tiles, num_tiles),
                          (img_size, img_size),
                          interpolation=cv2.INTER_LINEAR)
    binary = (upscaled > (upscaled.mean() + upscaled.std())).astype(np.uint8)

    # conver the bounding box into a square bounding box
    props = regionprops(binary)
    if len(props) > 0:
        bbox = regionprops(binary)[0].bbox
        square_bbox = data_utils.get_square_bbox(bbox, img_size=img_size, pad=pad)
    else:
        square_bbox = [0, 0, img_size, img_size]

    crop_coords = {
        'h_0': int(square_bbox[0]),
        'h_1': int(square_bbox[2]),
        'w_0': int(square_bbox[1]),
        'w_1': int(square_bbox[3]),
    }
    return crop_coords


def generate_cropped_coords(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # load model and train config
    model = model_utils.load_model(**config['model']).eval()
    model.to(device)

    config_filepath = os.path.join(config['model']['ckpt_dir'].rstrip('/').split('fold')[0],
                                   'config.json')
    train_config = utils.load_config_from_yaml(config_filepath)

    img_version = train_config['input']['img_version']
    img_size = int(img_version.split('x')[0])
    tile_size = train_config['data']['params']['tile_size']
    pad = config['default'].get('pad', 0)
    output_dir = os.path.join(config['output']['output_dir'], f'{img_version}_{tile_size}_{pad}')

    # cv folds and image stats
    cv_folds_dir = train_config['input']['cv_folds']
    img_stats = data_utils.load_img_stats(os.path.join(cv_folds_dir, img_version), 0)
    norm_cols = train_config['data'].get('norm_cols')

    df_mela = data_utils.load_data(config['input']['train'],
                                   duplicate_path=config['input'].get('duplicates'),
                                   cv_folds_dir=cv_folds_dir,
                                   external_filepaths=config['input'].get('external_filepaths'),
                                   image_map=config['input'].get('image_map'),
                                   keep_prob=config['default'].get('keep_prob', 1))

    df_test = data_utils.load_data(config['input']['test'],
                                   image_map=config['input'].get('image_map'),
                                   keep_prob=config['default'].get('keep_prob', 1))
    df_test['fold'] = 'test'
    df_test['target'] = None

    df_mela = pd.concat((df_mela, df_test), axis=0, ignore_index=True)
    df_mela['image_dir'] = [
        os.path.join(row.image_dir, 'test') if row.fold == 'test' else os.path.join(row.image_dir, 'train')
        for row
        in df_mela.itertuples()
    ]
    df_mela['source_group'] = df_mela['image_dir'].apply(lambda x: x.rstrip('/').split('/')[-3])

    for source in df_mela['source_group'].unique():
        output_source_dir = os.path.join(output_dir, source)
        for group in ['train', 'test']:
            output_group_dir = os.path.join(output_source_dir, group)
            if not os.path.exists(output_group_dir):
                print(f'Generating output dir {output_group_dir}')
                os.makedirs(output_group_dir)

    # save config to disk
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # augmentor and dataset
    aug, _, _ = train_utils.get_augmentors(transforms={'normalize': img_stats},
                                           norm_cols=norm_cols)
    dataset = train_utils.get_dataset('tile',
                                      df_mela,
                                      augmentor=aug,
                                      **train_config['data']['params'])

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        x, y = dataset[i]
        df_idx = dataset.df.iloc[i]
        if use_cuda:
            x = x.cuda()
        with torch.no_grad():
            y_pred = model(x[None,...])
        weights = model.weights.detach().cpu().numpy()
        crop_coords = get_crop_coords(weights, img_size=img_size, pad=pad)
        group = 'test' if df_idx['fold'] == 'test' else 'train'
        with open(os.path.join(output_dir, df_idx['source_group'], group, f"{df_idx['image_name']}.json"), 'w') as f:
            json.dump(crop_coords, f)

