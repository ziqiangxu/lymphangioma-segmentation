"""
@Author: Daryl Xu <ziqiang_xu@qq.com>
"""
import os
import re
from typing import List
from decimal import Decimal

import numpy as np
import pydicom
import cv2 as cv


class Pixel:
    """
    volume存储顺序：height/dicom堆叠方向，row/x，column/y
    """
    def __init__(self, row: int, col: int, height: int = None):
        # 笛卡尔（左手）坐标系
        self.row = row  # y
        self.col = col  # x
        self.height = height  # z

    def get_eight_neighborhood(self, img: np.ndarray):
        """
        对于2D图像
        获取八邻域
        p1 p2 p3
        p4 p  p5
        p6 p7 p8
        """
        assert img.ndim == 2
        row = self.row
        col = self.col

        # 如果是边缘的点，则返回None
        if row == 0 or col == 0 or \
                row == (img.shape[0] - 1) or col == (img.shape[1] - 1):
            return []

        p1 = Pixel(row - 1, col - 1)
        p2 = Pixel(row - 1, col)
        p3 = Pixel(row - 1, col + 1)
        p4 = Pixel(row, col - 1)
        p5 = Pixel(row, col + 1)
        p6 = Pixel(row + 1, col - 1)
        p7 = Pixel(row + 1, col)
        p8 = Pixel(row + 1, col + 1)
        return p1, p2, p3, p4, p5, p6, p7, p8

    def get_26_neighborhood_3d(self, img: np.ndarray):
        assert img.ndim == 3
        row = self.row
        col = self.col
        height = self.height

        shape = img.shape
        # 不为在边缘的体素获取领域
        if height == 0 or row == 0 or col == 0 or \
                height == (shape[0] - 1) or \
                row == (shape[1] - 1) or \
                col == (shape[2] - 1):
            return []

        pixels = []
        # 第一层
        pixels1_center = Pixel(row, col, height)
        pixels.append(pixels1_center)
        pixels1_8 = self.get_eight_neighborhood(img[height - 1])
        for p in pixels1_8:
            p.height = height - 1
        pixels.extend(pixels1_8)

        # 中间层
        pixels2_8 = self.get_eight_neighborhood(img[height])
        for p in pixels2_8:
            p.height = height
        pixels.extend(pixels2_8)

        # 第三层
        pixels3_center = Pixel(row, col, height + 1)
        pixels.append(pixels3_center)
        pixels3_8 = self.get_eight_neighborhood(img[height + 1])
        for p in pixels3_8:
            p.height = height + 1
        pixels.extend(pixels3_8)

        return pixels

    def get_neighborhood_3d_arr(self, img: np.ndarray, half_width: int = 3,
                                half_height: int = 1) -> np.ndarray:
        def valid_index(test_value, upper):
            assert 0 <= test_value <= upper

        shape = img.shape
        height1 = self.height - half_height
        height2 = self.height + half_height + 1
        valid_index(height1, shape[0])
        valid_index(height2, shape[0])

        row1 = self.row - half_width
        row2 = self.row + half_width + 1
        valid_index(row1, shape[1])
        valid_index(row2, shape[1])

        column1 = self.col - half_width
        column2 = self.col + half_width + 1
        valid_index(column1, shape[2])
        valid_index(column2, shape[2])
        arr: np.ndarray = img[height1:height2, row1:row2, column1:column2]
        return arr.copy()

    def get_pixel(self, img: np.ndarray):
        assert img.ndim == 2
        return img[self.row, self.col]

    def get_pixel_3d(self, img: np.ndarray):
        assert img.ndim == 3
        return img[self.height, self.row, self.col]

    def get_slice(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3
        return img[self.height]

    def set_slice(self, img: np.ndarray, slice_: np.ndarray):
        assert img.ndim == 3
        img[self.height] = slice_

    def mark(self, mask: np.ndarray, mark_value: int = 1):
        """
        会更新mask参数
        :param mask:
        :param mark_value:
        :return:
        """
        if mask.ndim == 2:
            mask[self.row, self.col] = mark_value
        else:
            mask[self.height, self.row, self.col] = mark_value

    def __str__(self):
        return f'({self.row}, {self.col}, {self.height})'

    @staticmethod
    def print(pixels: []):
        """
        格式化输出
        """
        for p in pixels:
            print(f'Pixel{str(p)}')


class DcmLoadingException(Exception):
    def __init__(self, msg, *args,**kwargs):
        super(DcmLoadingException, self).__init__(args, kwargs)


def get_slice_location(path: str) -> float:
    """
    Get stack position from the number
    :param path:
    :return:
    """

    dcm = pydicom.dcmread(path, force=True)
    # return dcm.InStackPositionNumber
    return float(dcm.SliceLocation)


def load_dcm_series(files: List[str]):
    """
    Check and load the given DICOM files
    :param files:
    :return:
    """
    volume = []
    files.sort(key=get_slice_location)
    for file in files:
        dcm = pydicom.dcmread(file, force=True)
        if not dcm.file_meta.get('TransferSyntaxUID'):
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        volume.append(dcm.pixel_array)
    return files, np.stack(volume)


def _load_volume_from_jpg(files: List[str]) -> np.ndarray:
    """
    Load volume from jpg
    :param files:
    :return:
    """
    volume = []
    for file in files:
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        volume.append(img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
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
    sort the files by name
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
        base_name = os.path.basename(name)
        nums = pattern.findall(base_name)
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


def load_jpg_series(files: List[str]):
    """
    Load the jpg files
    :param files: 
    :return: 
    """
    sort_files_by_name(files)
    volume = _load_volume_from_jpg(files)
    return files, volume


def get_voxel_size(path: str) -> float:
    """
    Get the voxel size of the DICOM file
    :param path:
    :return:
    """
    dcm = pydicom.dcmread(path, force=True)
    x_str, y_str = dcm.PixelSpacing
    x = Decimal(str(x_str))
    y = Decimal(str(y_str))
    z = Decimal(str(dcm.SpacingBetweenSlices))
    print(float(x * y * z))
    return float(x * y * z)
