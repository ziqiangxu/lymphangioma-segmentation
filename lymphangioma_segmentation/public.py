"""
@Author: Daryl Xu <ziqiang_xu@qq.com>
"""
import os
from typing import List
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import nibabel as nib
try:
    import cv2 as cv
except ModuleNotFoundError:
    print('don\'t worry, cv2 is required just in development')
from lymphangioma_segmentation.image import Pixel


# TODO move this to config file
LOG_PATH = '/home/daryl/git/lymphangioma-segmentation/lymphangioma_segmentation/tmp/logs'


def img_show(img, size=(6, 6), output='tmp/tmp.png'):
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=size)
    axes.imshow(img, cmap='gray')
    figure.savefig(output)
    return figure, axes


def draw_mask_contours(img, mask, path, size=(6, 6), title: str = ""):
    """
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#what-are-contours
    此链接的文档有问题啊(findContours只返回结果是两个元素的元祖)
    将mask的边缘绘制到图像上
    :param img:
    :param mask:
    :param path:
    :param size:
    :param title:
    :return:
    """
    figure: Figure
    axis: Axes
    figure, axis = plt.subplots(figsize=size)
    axis.set_title(title)
    axis.imshow(img, cmap='gray')

    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours1 = np.array(contours[0])
    contours1 = contours1.reshape([int(contours1.size / 2), 2])

    codes = [Path.MOVETO]
    res = [Path.LINETO for i in range(contours1.shape[0] - 1)]
    codes.extend(res)

    contours_path = Path(contours1, codes)
    patch = patches.PathPatch(contours_path, color='red', lw=2, fill=False)
    axis.add_patch(patch)
    figure.savefig(path)
    return figure, axis


def show_seed(slice_: np.ndarray, seed: Pixel, path: str = None, size=(6, 6), title=''):
    """
    :param slice_:
    :param seed:
    :param path:
    :param size:
    :param title:
    :return:
    """
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=size)
    axes.imshow(slice_, cmap='gray')
    axes.set_title(title)
    rect = plt.Rectangle((seed.col, seed.row), 10, 10)
    axes.add_patch(rect)
    plt.show()
    if path:
        figure.savefig(path)


def save_img(img, path, size=(6, 6)):
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=size)
    axes.imshow(img, cmap='gray')
    figure.savefig(path)
    plt.close(figure)


def draw_curve(x_data, y_data, output=None, size=(6, 6), x_label='', y_label=''):
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=size)
    axes.plot(x_data, y_data)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    if output:
        figure.savefig(output)
        plt.close(figure)
    return figure, axes


def save_nii(img: np.ndarray, path: str, affine: np.ndarray = None):
    if affine is None:
        affine = np.eye(4)
    img_obj = nib.nifti1.Nifti1Image(img, affine)
    nib.save(img_obj, path)


def get_log_path(relative_path: str, create_dir=False):
    absolute_path = os.path.join(LOG_PATH, relative_path)
    if create_dir:
        os.makedirs(absolute_path, exist_ok=True)
    return absolute_path


def load_volume_from_jpg(files: List[str]) -> np.ndarray:
    """
    Load volume from jpg
    :param files:
    :return:
    """
    volume = []
    for file in files:
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        volume.append(img)
        plt.imshow(img, cmap='gray')
        plt.show()
    volume = np.stack(volume)
    volume = volume / volume.max() * 1024
    return volume


# Type1
# MR Neck PS.eT2W_SPAIR SENSE.Se 602.Img 28-32.jpg
# MR Neck PS.eT2W_SPAIR SENSE.Se 602.Img 29-32.jpg
# Type2

pattern = re.compile(r'[0-9]+')


def sort_files_by_name(names: List[str], reverse: bool = False):
    """
    sort the names by
    :param names:
    :param reverse:
    :return:
    """
    num_sort_index: int
    num_count: int

    def get_image_index(name: str):
        """
        Get the index of name.
        filename: 'MR Neck PS.eT2W_SPAIR SENSE.Se 602.Img 10-32.jpg'
        the index is 10
        :param name:
        :return:
        """
        nums = pattern.findall(name)
        if len(nums) != num_count:
            raise BaseException(f"can't exact index from the string: {name}")
        return float(nums[num_sort_index])

    if len(names) > 2:
        num1 = pattern.findall(os.path.basename(names[0]))
        num2 = pattern.findall(os.path.basename(names[1]))
        # 解析出来的数字数目应该一样多
        num_count = len(num1)
        assert num_count == len(num2)
        arr1 = np.array(num1)
        arr2 = np.array(num2)
        diff: np.ndarray = arr1 == arr2

        # 按道理最多只能有一个数字不一样
        # 40806068_20200827_MR_6_2_2.jpg
        # 40806068_20200827_MR_6_3_3.jpg
        # assert diff.sum() + 1 == num_count

        # numpy数组中: True = 1, False = 0
        num_sort_index = diff.argmin()
        # TODO remove this line
        # print(num1, num2, num_sort_index)

    names.sort(key=get_image_index, reverse=reverse)
    return names
