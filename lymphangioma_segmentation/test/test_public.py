"""
@Author: Daryl Xu <ziqiang_xu@qq.com>
"""
import os
import re

import numpy as np

from lymphangioma_segmentation import public
# pattern = re.compile(r'.*Img ([0-9]+)-.*')


def test_load_volume_from_jpg():
    # test_dir = '/home/daryl/git/medical-image-viewer/data_/LM/LM/64760252'
    test_dir = '/home/daryl/git/medical-image-viewer/data_/LM/LM/64744562'
    test_dir = '/home/daryl/git/lymphangioma-segmentation/test-data/LM/40806068'
    names_org = os.listdir(test_dir)
    names = names_org.copy()
    public.sort_files_by_name(names)
    for i in range(len(names)):
        print(names[i], names_org[i])
    # volume = public.load_volume_from_jpg(files)
    # np.save('64760252.npy', volume)
    # np.save('64744562.npy', volume)


if __name__ == '__main__':
    test_load_volume_from_jpg()
