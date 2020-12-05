"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import numpy as np
import matplotlib.pyplot as plt

from lymphangioma_segmentation import segmentation
from lymphangioma_segmentation.image import Pixel

# threshold 850-1200
DATA1 = 'test1.npy'
SEEDS1 = list()
SEEDS1.append(Pixel(220, 350, 11))
SEEDS1.append(Pixel(220, 340, 10))
SEEDS1.append(Pixel(220, 350, 9))
SEEDS1.append(Pixel(220, 350, 8))
SEEDS1.append(Pixel(220, 350, 7))
SEEDS1.append(Pixel(220, 350, 6))

# threshold 800-1200
PATIENT0 = 'p0.npy'
SEEDS_P0 = list()
SEEDS_P0.append(Pixel(324, 180, 9))
SEEDS_P0.append(Pixel(340, 200, 10))
SEEDS_P0.append(Pixel(340, 200, 11))
SEEDS_P0.append(Pixel(350, 190, 12))
SEEDS_P0.append(Pixel(350, 200, 13))
SEEDS_P0.append(Pixel(342, 180, 14))
SEEDS_P0.append(Pixel(342, 170, 15))

PATIENT1 = 'p1.npy'
SEEDS_P1 = list()


def test_optimized_thresholds():
    def compute(img, seed):
        neighborhood_arr = seed.get_neighborhood_3d_arr(img, 10)
        reference_intensity = neighborhood_arr.mean()
        slice_ = seed.get_slice(img)
        optimized_threshold = segmentation.get_optimized_threshold(
            slice_, seed, reference_intensity, 0.1, True)
        print(f'threshold: {optimized_threshold[0]}, '
              f'trigger ratio: {optimized_threshold[1]}')

    # npy = np.load(DATA1)
    # for s in SEEDS1:
    #     compute(npy, s)

    npy0 = np.load(PATIENT0)
    for s in SEEDS_P0:
        compute(npy0, s)


def test_show_seeds():
    img = np.load(PATIENT0)
    # for i in range(img.shape[0]):
    for s in SEEDS_P0:
        slice_ = img[s.height]
        figure, axes = plt.subplots()
        # figure.canvas.set_window_title(f'图片索引：{i}')
        axes.imshow(slice_, cmap='gray')

        rec = plt.Rectangle((s.col, s.row), 10, 10)
        axes.add_patch(rec)

        plt.show()


if __name__ == '__main__':
    # test_optimized_thresholds()
    test_show_seeds()
