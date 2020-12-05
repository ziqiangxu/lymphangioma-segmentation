"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import numpy as np


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
            assert 0 <= test_value < upper

        shape = img.shape
        height1 = self.height - half_height
        height2 = self.height + half_height
        valid_index(height1, shape[0])
        valid_index(height2, shape[0])

        row1 = self.row - half_width
        row2 = self.row + half_width
        valid_index(row1, shape[1])
        valid_index(row2, shape[1])

        column1 = self.col - half_width
        column2 = self.col + half_width
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

    def get_slice(self, img: np.ndarray):
        assert img.ndim == 3
        return img[self.height]

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
