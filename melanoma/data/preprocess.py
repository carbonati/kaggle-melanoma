import os
import json
import numpy as np
import pandas as pd
import cv2
from zipfile import ZipFile
from tqdm import tqdm

from utils import data_utils


def preprocess_images(root,
                      data_dir,
                      output_dir,
                      size,
                      interpolation=4,
                      img_format='jpg',
                      quality=100,
                      return_meta=True,
                      duplicate_path=None):

    compression = cv2.IMWRITE_PNG_COMPRESSION if img_format == 'png' else cv2.IMWRITE_JPEG_QUALITY
    train_image_dir = os.path.join(root, 'train')
    test_image_dir = os.path.join(root, 'test')

    if size is not None:
        base_output_dir = os.path.join(output_dir, f'{size}x{size}_{img_format}_{quality}_{interpolation}')
        train_output_path = os.path.join(base_output_dir, 'train.zip')
        test_output_path = os.path.join(base_output_dir, 'test.zip')
    else:
        base_output_dir = os.path.join(output_dir, f'original_{interpolation}')
        train_output_dir = None
        test_output_dir = None

    meta_dir = os.path.join(base_output_dir, 'metadata')
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    df_train = data_utils.load_data(os.path.join(data_dir, 'train.csv'), duplicate_path=duplicate_path)
    df_train['image_dir'] = train_image_dir
    df_train['partition'] = 'train'
    df_test = data_utils.load_data(os.path.join(data_dir, 'test.csv'))
    df_test['image_dir'] = test_image_dir
    df_test['partition'] = 'test'

    df_mela = pd.concat(
        (df_train[['image_name', 'image_dir', 'partition']], df_test[['image_name', 'image_dir', 'partition']]),
        axis=0,
        ignore_index=True
    )

    print(f'Saving train images to {train_output_path}')
    print(f'Saving test images to {test_output_path}')
    if return_meta:
        print(f'Saving metadata to {meta_dir}')
    with ZipFile(train_output_path, 'w') as train_file, ZipFile(test_output_path, 'w') as test_file:
        for row in tqdm(df_mela.itertuples(), total=len(df_mela), desc='Preprocessing images'):
            image_id = row.image_name
            img = data_utils.load_image(row.image_dir, image_id)
            if size is None:
                img = data_utils.trim_img(img)
                meta = data_utils.get_img_stats(img)
            else:
                img, meta = data_utils.resize_img(img,
                                                  size=size,
                                                  interpolation=interpolation)

                img = cv2.imencode(f'.{img_format}', img, [compression, quality])[1]
                if row.partition == 'train':
                    train_file.writestr(f'{image_id}.{img_format}', img)
                else:
                    test_file.writestr(f'{image_id}.{img_format}', img)

            if return_meta:
                with open(os.path.join(meta_dir, f'{image_id}.json'), 'w') as f:
                    json.dump(str(meta), f)

