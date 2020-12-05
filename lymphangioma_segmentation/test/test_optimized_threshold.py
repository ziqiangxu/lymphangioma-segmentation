"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import public

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
SEEDS_P1.append(Pixel(200, 182, 2))
SEEDS_P1.append(Pixel(212, 165, 3))
SEEDS_P1.append(Pixel(227, 156, 4))
SEEDS_P1.append(Pixel(223, 159, 5))
SEEDS_P1.append(Pixel(237, 165, 6))
SEEDS_P1.append(Pixel(256, 164, 7))
SEEDS_P1.append(Pixel(272, 172, 8))
SEEDS_P1.append(Pixel(277, 181, 9))
SEEDS_P1.append(Pixel(276, 182, 10))
SEEDS_P1.append(Pixel(295, 170, 11))


def test_optimized_thresholds():
    def compute(img, seed):
        neighborhood_arr = seed.get_neighborhood_3d_arr(img, 10)
        reference_intensity = neighborhood_arr.mean()
        slice_ = seed.get_slice(img)
        optimized_threshold, trigger_ratio = segmentation.get_optimized_threshold(
            slice_, seed, reference_intensity, 0.5, True)
        print(f'threshold: {optimized_threshold}, '
              f'trigger ratio: {trigger_ratio}')
        mask = segmentation.region_grow(slice_, seed, optimized_threshold)
        save_path = os.path.join(log_path, f'mask_contours_{seed.height}.png')
        public.draw_mask_contours(slice_, mask, save_path)

    # npy = np.load(DATA1)
    # log_path = public.get_log_path('data1', True)
    # for s in SEEDS1:
    #     compute(npy, s)

    # npy0 = np.load(PATIENT0)
    # log_path = public.get_log_path('patient0', True)
    # for s in SEEDS_P0:
    #     compute(npy0, s)

    npy1 = np.load(PATIENT1)
    log_path = public.get_log_path('patient1', True)
    for s in SEEDS_P1:
        compute(npy1, s)

    # npy2 = np.load(PATIENT2)
    # log_path = public.get_log_path('patient2', True)
    # for s in SEEDS_P2:
    #     compute(npy2, s)


def test_show_seeds():
    img: np.ndarray = np.load(PATIENT0)
    for s in SEEDS_P0:
        slice_: np.ndarray = img[s.height]
        figure, axes = plt.subplots()
        axes.imshow(slice_, cmap='gray')
        rec = plt.Rectangle((s.col, s.row), 10, 10)
        axes.add_patch(rec)

        plt.show()


def test_display_volume():
    img: np.ndarray = np.load(PATIENT1)
    for i in range(img.shape[0]):
        slice_: np.ndarray = img[i]
        figure, axes = plt.subplots()
        figure.canvas.set_window_title(f'图片索引：{i}')
        axes.imshow(slice_, cmap='gray')
        plt.show()


if __name__ == '__main__':
    test_optimized_thresholds()
    # test_show_seeds()
    # test_display_volume()
