"""
Author: Daryl Xu
E-mail: ziqiang_xu@qq.com
"""
import os
import re

import numpy as np

from lymphangioma_segmentation import public

# 'MR Neck PS.eT2W_SPAIR SENSE.Se 602.Img 10-32.jpg'
pattern = re.compile(f'.*Img ([0-9]+)-.*')


def get_image_index(path: str):
    """
    Get the index of image.
    filename: 'MR Neck PS.eT2W_SPAIR SENSE.Se 602.Img 10-32.jpg'
    the index is 10
    :param path:
    :return:
    """
    filename = os.path.basename(path)
    res = pattern.match(filename)
    if not res:
        raise BaseException("can't exact index from filename")
    return float(res[1])


def test_load_volume_from_jpg():
    # test_dir = '/home/daryl/git/medical-image-viewer/data_/LM/LM/64760252'
    test_dir = '/home/daryl/git/medical-image-viewer/data_/LM/LM/64744562'

    files = os.listdir(test_dir)
    files.sort(key=get_image_index)

    for i in range(len(files)):
        files[i] = os.path.join(test_dir, files[i])

    volume = public.load_volume_from_jpg(files)
    # np.save('64760252.npy', volume)
    np.save('64744562.npy', volume)


if __name__ == '__main__':
    test_load_volume_from_jpg()
