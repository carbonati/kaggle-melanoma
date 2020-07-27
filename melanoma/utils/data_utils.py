import os
import numpy as np
import pandas as pd
import cv2


def load_data(filepath,
              duplicate_path=None):
    df_mela = pd.read_csv(filepath)

    # need to add more logic here for the test set
    if duplicate_path is not None:
        df_dupes = pd.read_csv(duplicate_path)
        image_ids_duped = df_dupes['ISIC_id_paired'].tolist()
        df_mela = df_mela.loc[df_mela['image_name'].isin(image_ids_duped)].reset_index(drop=True)

    return df_mela


def load_image(root, image_id, bgr2rgb=True):
    img = cv2.imread(os.path.join(root, f'{image_id}.jpg'))
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_img_stats(img):
    x = (img.reshape(-1, 3) - 255).astype(np.float32) / 255.
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
