"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from lymphangioma_segmentation import segmentation
from lymphangioma_segmentation.image import Pixel
from lymphangioma_segmentation import public
from lymphangioma_segmentation.log import get_logger

# threshold 850-1200
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DATA1 = 'data/test1.npy'
SEEDS1 = list()
SEEDS1.append(Pixel(220, 350, 11))
SEEDS1.append(Pixel(220, 340, 10))
SEEDS1.append(Pixel(220, 350, 9))
SEEDS1.append(Pixel(220, 350, 8))
SEEDS1.append(Pixel(220, 350, 7))
SEEDS1.append(Pixel(220, 350, 6))

# threshold 800-1200
PATIENT0 = 'data/p0.npy'
SEEDS_P0 = list()
SEEDS_P0.append(Pixel(324, 180, 9))
SEEDS_P0.append(Pixel(340, 200, 10))
SEEDS_P0.append(Pixel(340, 200, 11))
SEEDS_P0.append(Pixel(350, 190, 12))
SEEDS_P0.append(Pixel(350, 200, 13))
SEEDS_P0.append(Pixel(342, 180, 14))
SEEDS_P0.append(Pixel(342, 170, 15))

PATIENT1 = 'data/p1.npy'
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

# 这批数据来自于jpg图像
P_64744562 = 'data/64744562.npy'
P_64760252 = 'data/64760252.npy'
seeds_dict = {
    P_64744562: [
        Pixel(186, 95, 8),
        Pixel(193, 104, 10),
        Pixel(199, 121, 11),
        Pixel(199, 121, 12),
        Pixel(199, 121, 13),
        Pixel(199, 121, 14),
        Pixel(199, 121, 15),
        Pixel(199, 121, 16),
        Pixel(199, 121, 17),
        Pixel(199, 121, 18),
        Pixel(199, 121, 19),
        Pixel(247, 132, 21)
    ],
    P_64760252: [
        Pixel(357, 347, 3),
        Pixel(352, 358, 4),
        Pixel(332, 354, 5),
        Pixel(333, 349, 6),
        Pixel(333, 349, 7),
        Pixel(333, 349, 8),
    ],
    PATIENT0: SEEDS_P0,
    PATIENT1: SEEDS_P1,
    DATA1: SEEDS1
}


def get_test_data(data_id: str) -> Tuple[np.ndarray, List[Pixel]]:
    img: np.ndarray = np.load(data_id)
    seeds = seeds_dict[data_id]
    return img, seeds


def test_optimized_thresholds():
    thresholds = []
    ratios = []

    def compute(img, seed):
        neighborhood_arr = seed.get_neighborhood_3d_arr(img, 10)
        reference_intensity = neighborhood_arr.mean()
        slice_ = seed.get_slice(img)
        optimal_threshold, trigger_ratio = segmentation.get_optimized_threshold(
            slice_, seed, reference_intensity, 3, False)

        thresholds.append(optimal_threshold)
        ratios.append(trigger_ratio)
        logger.debug(f'threshold: {optimal_threshold}, trigger ratio: {trigger_ratio}')

        mask = segmentation.region_grow(slice_, seed, optimal_threshold)
        save_path = os.path.join(log_path, f'mask_contours_{seed.height}.png')
        public.draw_mask_contours(slice_, mask, save_path,
                                  title=f't: {optimal_threshold:.1f}|{seed.height}')
        return optimal_threshold, trigger_ratio

    # npy = np.load(DATA1)
    # log_path = public.get_log_path('data1', True)
    # for s in SEEDS1:
    #     compute(npy, s)

    # npy0 = np.load(PATIENT0)
    # log_path = public.get_log_path('patient0', True)
    # for s in SEEDS_P0:
    #     compute(npy0, s)

    # npy1 = np.load(PATIENT1)
    npy1, seeds = get_test_data(test_data_id)
    log_path = public.get_log_path('patient1', True)
    for s in seeds:
        compute(npy1, s)

    # npy2 = np.load(PATIENT2)
    # log_path = public.get_log_path('patient2', True)
    # for s in SEEDS_P2:
    #     compute(npy2, s)
    thresholds = np.array(thresholds)
    logger.debug(f'threshold mean: {thresholds.mean()}')

    a_annotation: int
    figure: Figure
    axis1: Axes
    axis2: Axes
    figure, (axis1, axis2) = plt.subplots(1, 2)
    axis1.plot(thresholds)
    axis1.set_title('optimal thresholds')
    axis1.text(0, thresholds.max(), f'mean: {thresholds.mean(): .2f}\n'
                                    f'SE: {thresholds.std(): .2f}')

    axis2.plot(ratios)
    axis2.set_title('trigger ratio')
    plt.show()


def test_grow_slice():
    img, _ = get_test_data(test_data_id)
    seed = Pixel(199, 121, 17)
    slice_ = seed.get_slice(img)
    intensity = seed.get_pixel(slice_)
    segmentation.get_optimized_threshold(slice_, seed, intensity, 8, True)


def test_show_seeds():
    img, seeds = get_test_data(test_data_id)
    for s in seeds:
        slice_: np.ndarray = img[s.height]
        figure, axes = plt.subplots()
        axes.imshow(slice_, cmap='gray')
        rec = plt.Rectangle((s.col, s.row), 10, 10)
        axes.add_patch(rec)

        plt.show()


def test_grow_every_slice():
    img, seeds = get_test_data(test_data_id)
    seed_first = seeds[0]

    # img: np.ndarray = np.load(PATIENT0)
    # seed_first = SEEDS_P0[3]

    # img: np.ndarray = np.load(PATIENT1)
    # seed_first = SEEDS_P1[3]

    display = True

    seed = seed_first
    slice_ = seed.get_slice(img)
    seed_region_intensity = seed.get_neighborhood_3d_arr(img, 5).mean()

    optimized_threshold, trigger_ratio = segmentation.get_optimized_threshold(
        slice_, seed, seed_region_intensity, 0.5, True)
    mask_res = segmentation.region_grow(slice_, seed, optimized_threshold).astype(np.float)

    if display:
        plt.imshow(mask_res)
        plt.show()

    mask_3d = np.full(img.shape, np.nan)
    seed.set_slice(mask_3d, mask_res)

    optimized_thresholds = [optimized_threshold]

    def get_mask_mean_std():
        mask_3d_tmp = (mask_3d * img).astype(np.float)
        print(mask_3d_tmp.dtype, img.dtype, mask_3d.dtype)
        mask_3d_tmp[mask_3d_tmp == 0] = np.nan
        m, s = np.nanmean(mask_3d_tmp), np.nanstd(mask_3d_tmp)
        return float(m), float(s)

    while seed.height < img.shape[0] - 1:
        mean, std = get_mask_mean_std()
        try:
            seed, slice_next = segmentation.get_seed_in_neighbor_slice(seed, img, mask_res, mean, std, True)
        except Exception as e:
            e.with_traceback()
        if seed is None:
            # 停止层间生长
            break

        seed_region_intensity = seed.get_pixel(slice_next)
        optimized_threshold, _ = segmentation.get_optimized_threshold(slice_next, seed,
                                                                      seed_region_intensity, 0.5, True)
        upper = mean + 3 * std
        lower = mean - 3 * std
        if not lower < optimized_threshold < upper:
            print(f'range: ({lower}, {upper}), threshold: {optimized_threshold}')
            break

        optimized_thresholds.append(optimized_threshold)
        mask_res = segmentation.region_grow(slice_next, seed, optimized_threshold)
        seed.set_slice(mask_3d, mask_res)

        if display:
            plt.imshow(mask_res)
            plt.show()

    seed = seed_first
    while seed.height > 0:
        mean, std = get_mask_mean_std()
        seed, slice_next = segmentation.get_seed_in_neighbor_slice(seed, img, mask_res, mean, std, False)
        if seed is None:
            # 停止层间生长
            break

        seed_region_intensity = seed.get_pixel(slice_next)
        optimized_threshold, _ = segmentation.get_optimized_threshold(slice_next, seed,
                                                                      seed_region_intensity, 0.5, True)
        upper = mean + 3 * std
        lower = mean - 3 * std
        if not lower < optimized_threshold < upper:
            print(f'range: ({lower}, {upper}), threshold: {optimized_threshold}')
            break

        optimized_thresholds.append(optimized_threshold)
        mask_res = segmentation.region_grow(slice_next, seed, optimized_threshold)
        seed.set_slice(mask_3d, mask_res)

        thresholds_arr = np.array(optimized_thresholds)
        print(f'mean of threshold: {thresholds_arr.mean()}, {thresholds_arr}')
        if display:
            plt.imshow(mask_res)
            plt.show()
        public.save_nii(img, 'tmp/img.nii.gz')
        mask_3d[np.isnan(mask_3d)] = 0
        mask_3d = mask_3d.astype(np.uint8)
        public.save_nii(mask_3d, 'tmp/mask.nii.gz')


def test_grow_every_slice1():
    img: np.ndarray = np.load(PATIENT1)
    mask_3d, mean, std = segmentation.grow_by_every_slice(SEEDS_P1[3], img)
    logger.debug(f'mean: {mean}, std: {std}')
    public.save_nii(mask_3d, 'tmp/mask_3d.nii.gz')


def test_display_volume():
    img, _ = get_test_data(test_data_id)
    for i in range(img.shape[0]):
        slice_: np.ndarray = img[i]
        figure, axes = plt.subplots()
        figure.canvas.set_window_title(f'图片索引：{i}')
        axes.imshow(slice_, cmap='gray')
        plt.show()


def test_statistic_threshold():
    img: np.ndarray = np.load(DATA1)
    slice_ = img[8]

    test_rect: np.ndarray = slice_[223: 245, 320:334]
    hist, edges = np.histogram(test_rect, 5)

    figure: Figure
    axis1: Axes
    axis2: Axes
    figure, (axis1, axis2) = plt.subplots(1, 2)
    axis1.imshow(test_rect, cmap='gray')
    axis2.hist(test_rect, bins=edges)
    # axis2.imshow(test_rect)
    plt.show()


def test_grow_slice_grow_3d():
    """
    先通过2d生长统计出最优的阈值
    再使用3d区域生长进行分割
    :return:
    """
    img, seeds = get_test_data(test_data_id)
    seed_first = seeds[0]
    _, mean, std = segmentation.grow_by_every_slice(seed_first, img, ratio=3)
    logger.debug(f'mean: {mean}, std: {std}')

    # threshold = mean + std
    threshold = mean - 0.8 * std
    # threshold = mean
    # threshold = mean + std * 0.8

    logger.debug(f'the threshold: {threshold}')
    mask_3d = segmentation.region_grow_3d(img, seed_first, threshold)
    public.save_nii(img, 'output/262.nii.gz')
    public.save_nii(mask_3d, 'output/262_mask.nii.gz')
    # for i in range(0, mask_3d.shape[0]):
    #     public.draw_mask_contours(img[i], mask_3d[i], 'tmp/t.png')
    #     plt.show()


def test_mip():
    img, _ = get_test_data(test_data_id)
    mip = img.max(0)
    plt.imshow(mip, cmap='gray')
    plt.show()


if __name__ == '__main__':
    logger = get_logger('lymphangioma_segmentation')
    # test_data_id = P_64744562
    # test_data_id = P_64760252
    test_data_id = DATA1
    # test_data_id = PATIENT0
    # test_data_id = PATIENT1
    # test_optimized_thresholds()
    # test_grow_slice()
    # test_show_seeds()
    # test_statistic_threshold()
    # test_display_volume()
    # test_grow_every_slice()
    # test_grow_every_slice1()
    test_grow_slice_grow_3d()
    os.system('source ~/.bashrc;itksnap -g ~/git/lymphangioma-segmentation/lymphangioma_segmentation/output/262.nii.gz -s ~/git/lymphangioma-segmentation/lymphangioma_segmentation/output/262_mask.nii.gz ')
    # test_mip()
