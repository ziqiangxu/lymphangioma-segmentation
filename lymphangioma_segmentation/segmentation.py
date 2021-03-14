"""
@Author: Daryl Xu <ziqiang_xu@qq.com>
"""
import math
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from numpy import ndarray

from lymphangioma_segmentation.image import Pixel
from lymphangioma_segmentation import public
from lymphangioma_segmentation.public import img_show, draw_curve, save_img, save_nii
from lymphangioma_segmentation.log import get_logger

logger = get_logger(__name__)


def get_optimized_threshold(img: np.ndarray, seed: Pixel, reference_intensity: float,
                            ratio: float = 3, verbose=False, min_iter: int = 3) -> Tuple[float, float]:
    """
    根据种子点计算获取最优的分割阈值
    基本思想：
        根据一定规则使阈值缓慢下降，当下降到边界值的时候，用这一阈值分割出来的区域面积会突然增长（错误地将非ROI区域也纳入了）
    :param img:
    :param seed:
    :param reference_intensity: 根据种子点周围的像素点统计出来的强度值，比直接用种子点强度更稳定
    :param ratio:
    :param verbose:
    :param min_iter:
    :return: threshold, ratio_
    """
    img = img.astype(np.float)
    if verbose:
        logger.debug(f'seed_value: {seed.get_pixel(img)}, max_value: {img.max()}')

    descent_rate = 0.95

    threshold = reference_intensity * descent_rate
    thresholds = [threshold, ]

    # 这个循环可以并行操作
    # 计算公式 (area3 - area2) / (area2 - area1) 大于2左右的一个经验值
    mask = region_grow(img, seed, threshold)
    area1 = mask.sum()

    def threshold_descent(current_threshold, mask_, iter_num):
        return current_threshold * descent_rate

        # 这种计算方法，求和的时候很容易就溢出了
        # reference_intensity = (mask * img).sum() / mask.sum()
        # TODO 直接将mask初始化成np.nan, np.full(img.shape, np.nan, dtype = np.uint8)
        # TODO 并验证一下速度是不是有提高
        selected = img * mask
        selected[selected == 0] = np.nan
        # TODO 为什么会出现selected全是nan的情况？PATIENT1
        previous_mean = np.nanmean(selected)
        threshold_based_previous = previous_mean * math.pow(descent_rate, iter_num)
        if threshold_based_previous < current_threshold:
            return threshold_based_previous
        else:
            # 有时候根据先前的分割面积计算出来的阈值可能比当前的阈值大，阈值就停止下降了
            # 这时就切换下降策略，直接根据当前阈值下降
            logger.debug('descent according to the previous threshold')
            return current_threshold * descent_rate

    threshold = threshold_descent(threshold, mask, 2)
    thresholds.append(threshold)

    mask = region_grow(img, seed, thresholds[1])
    area2 = mask.sum()

    threshold = threshold_descent(threshold, mask, 3)
    thresholds.append(threshold)

    mask = region_grow(img, seed, thresholds[2])
    area3 = mask.sum()
    areas = [area1, area2, area3]

    ratios = []
    iteration = 3

    def draw_log_pic():
        # threshold curve
        x = thresholds[:len(areas)]
        figure, axes = draw_curve(x, areas, x_label='threshold', y_label='area')
        axes.axvline(x=optimized_threshold, linestyle='-', color='red')
        if verbose:
            figure.savefig('tmp/area_threshold_curve.png')
        plt.close(figure)

        x = thresholds[:len(areas) - 1]
        figure, axes = draw_curve(x, areas[:-1], x_label='threshold', y_label='area')
        axes.axvline(x=optimized_threshold, linestyle='-', color='red')
        if verbose:
            figure.savefig('tmp/area_threshold_curve_.png')
        plt.close(figure)

        # area delta
        area_s = np.array(areas[:-1])
        area_g = np.array(areas[1:])
        area_delta = area_g - area_s
        index = np.linspace(1, area_delta.size, num=area_delta.size)
        if verbose:
            draw_curve(index, area_delta, 'tmp/area_delta.png', y_label='area_delta')
            draw_curve(index[:-1], area_delta[:-1], 'tmp/area_delta_.png',
                       y_label='area_delta')

        index = [i for i in range(len(ratios))]
        if verbose:
            draw_curve(index, ratios, 'tmp/ratios_curve.png', x_label='iter', y_label='ratios')
            draw_curve(index[:-1], ratios[:-1], 'tmp/ratios_curve_.png', x_label='iter', y_label='ratios')
            draw_curve(index[min_iter:], ratios[min_iter:], 'tmp/ratios_curve_min.png', y_label='ratios')
            draw_curve(index[min_iter:-1], ratios[min_iter:-1], 'tmp/ratios_curve_min_.png', y_label='ratios')

        index = [i for i in range(len(thresholds))]
        if verbose:
            draw_curve(index, thresholds, 'tmp/thresholds_curve.png', x_label='iter', y_label='threshold')
            draw_curve(index[:-1], thresholds[:-1], 'tmp/thresholds_curve_.png', x_label='iter', y_label='threshold')

    while True:
        # ratio_ = None
        if area2 != area1:
            ratio_ = (area3 - area2) / (area2 - area1)
            ratios.append(ratio_)
            if verbose:
                logger.debug(
                    f'ratio: {ratio_: .2f}, threshold: {threshold: .1f}, area3: {area3}, area2: {area2}, area1: {area1}')
            if ratio_ > ratio and iteration > min_iter:
                # if abs(threshold - 1100) < 100:
                if verbose:
                    save_img(mask * img, 'tmp/critical_mask.png')
                optimized_index = len(areas) - 2
                optimized_threshold = thresholds[optimized_index]
                logger.info(f'The optimized threshold: {optimized_threshold}')
                # show the optimized segmentation
                #             img_show(mask)
                draw_log_pic()
                return optimized_threshold, ratio_
        else:
            ratios.append(0)
        # 如果面积相等，需要打破这个状态，避免下降停止

        area1 = area2
        area2 = area3

        iteration += 1

        threshold = threshold_descent(threshold, mask, iteration)
        thresholds.append(threshold)

        mask = region_grow(img, seed, threshold)
        area3 = mask.sum()
        areas.append(area3)

        if iteration > 50:
            break

    ratios_valid = ratios[min_iter:]
    ratio_ = max(ratios_valid)
    index = ratios_valid.index(ratio_)
    optimized_threshold = thresholds[min_iter + index - 1]
    draw_log_pic()
    if verbose:
        # 遇到内层 中层 外层三层强度值差别很大,
        # 如内层在100附近,中层10附近,外层在100附近.
        # 从内层生长到0附近可能需要下降多次才能发生泄漏
        plt.plot(ratios)
        plt.show()
    return optimized_threshold, ratio_


#     if mask.sum() > area_limit:
#         break
#     logger.debug(f'threshold: {threshold}, area: {mask.sum()}')


def region_grow(img: np.ndarray, seed: Pixel, threshold: float) -> np.ndarray:
    """
    2D区域生长
    Region grow in 2D images
    :param img:
    :param seed:
    :param threshold:
    :return:
    """
    assert img.ndim == 2
    mask = np.zeros(img.shape, dtype=np.int8)  # 将相似的点标记为1
    # Pixel.print(neighborhoods)
    pixels_stack = [seed]
    while True:
        if len(pixels_stack) == 0:
            break
        cursor = pixels_stack.pop()
        neighborhoods = cursor.get_eight_neighborhood(img)

        for pixel in neighborhoods:
            flag = pixel.get_pixel(mask)
            pixel_value = pixel.get_pixel(img)
            if pixel_value > threshold and flag == 0:
                pixels_stack.append(pixel)
                pixel.mark(mask)
    return mask


def fine_tune_roi(roi: ndarray, image: ndarray, mask: ndarray):
    """
    基于已分割的图像进行微调
    Fine tune base on the segmented images
    :param roi:
    :param image:
    :param mask:
    :return:
    """
    seg = (image * mask).astype(np.float)
    seg[seg == 0] = np.nan
    mean = np.nanmean(seg)
    std = np.nanstd(seg)
    roi_mask: ndarray = roi > mean - 1.5 * std
    return roi_mask.astype(np.int8)

def fine_tune_roi1(roi: ndarray, threshold: float):
    """
    使用给定阈值对roi区域进行分割
    Segment roi by a given threshold
    :param roi:
    :param threshold:
    :return:
    """
    index = np.unravel_index(roi.argmax(), roi.shape)
    seed = Pixel(index[0], index[1])
    return region_grow(roi, seed, threshold)


def region_grow_3d(img: np.ndarray, seed: Pixel, threshold: float) -> np.ndarray:
    """
    :param img:
    :param seed:
    :param threshold:
    :return:
    """
    assert img.ndim == 3
    mask = np.zeros(img.shape, dtype=np.int8)
    pixels_stack = [seed]
    while len(pixels_stack):
        cursor = pixels_stack.pop()
        neighborhoods = cursor.get_26_neighborhood_3d(img)
        for pixel in neighborhoods:
            flag = pixel.get_pixel_3d(mask)
            pixel_value = pixel.get_pixel_3d(img)
            if pixel_value > threshold and flag == 0:
                pixels_stack.append(pixel)
                pixel.mark(mask)
    return mask


def region_grow_3d_without_threshold(img: np.ndarray, seed: Pixel) -> np.ndarray:
    """
    参考文献： https://kns.cnki.net/kcms/detail/detail.aspx?FileName=YXWZ201707002&DbName=CJFQ2017
    这种方法有严重的性能问题
    :param img:
    :param seed:
    :return:
    """
    seeds = seed.get_26_neighborhood_3d(img)

    # seeds.append(seed)

    def get_seeds_array():
        seeds_value = []
        for s in seeds:
            seeds_value.append(s.get_pixel_3d(img))
        return np.array(seeds_value)

    seeds_arr = get_seeds_array()
    seeds_mean = seeds_arr.mean()
    seeds_std = seeds_arr.std()
    logger.debug(f'seeds_mean: {seeds_mean}, seeds_std: {seeds_std}')

    def compare_pixel(value):
        if value > seeds_mean:
            return True
        # current_mean = (mask * img).mean() / mask.sum() * mask.size
        # if (seeds_mean - value) < 2 * seeds_std and (current_mean - value) < seeds_std:
        if (seeds_mean - value) < seeds_std:
            return True
        return False

    mask = np.zeros(img.shape, dtype=np.int8)
    while len(seeds):
        cursor = seeds.pop()
        neighborhoods = cursor.get_26_neighborhood_3d(img)
        logger.debug(f'pixels number: {mask.sum()}')
        for pixel in neighborhoods:
            flag = pixel.get_pixel_3d(mask)
            pixel_value = pixel.get_pixel_3d(img)
            if flag == 0 and compare_pixel(pixel_value):
                seeds.append(pixel)
                pixel.mark(mask)
    return mask


def get_seed_in_neighbor_slice(seed: Pixel, img: np.ndarray, mask: np.ndarray,
                               mean: float, std: float, increase: bool = True,
                               show_seed: bool = False
                               ) -> Tuple[Pixel, np.ndarray]:
    """
    获取相邻slice的种子点
    统计当前slice已分割的区域，如果相邻层的相同区域的最大像素值在当前slice已分割区域平均值附近，则将其作为种子点
    :param seed:
    :param img:
    :param mask: roi in current slice
    :param mean:
    :param std:
    :param increase: if True increase the height
    :param show_seed:
    :return:
    """
    assert img.ndim == 3
    if increase:
        height = seed.height + 1
    else:
        height = seed.height - 1
    seed_neighbor = Pixel(seed.row, seed.col, height)

    slice_ = seed.get_slice(img)
    slice_next: np.ndarray = seed_neighbor.get_slice(img)
    #
    # roi_region: np.ndarray = mask * slice_.astype(np.float)
    # roi_region[roi_region == 0] = np.nan
    # mean = np.nanmean(roi_region)
    # std = np.nanmean(roi_region)

    # 缩小mask的区域
    mask_ = mask.astype(np.float)
    mask_[slice_ < mean] = 0
    # 如果当前层的mask已经非常小了, 则停止生长, 防止下一层生长到全图
    if np.nansum(mask_) < 300:
        return None, slice_next

    seed_region_next: np.ndarray = slice_next * mask_

    # TODO 这个地方有问题，可能找到ROI之外. 特别是在上一层分割不太准确的时候, 所以需要着重提高区域生长的精度
    upper = mean + 2 * std
    lower = mean - std

    seed_region_next[seed_region_next < lower] = 0
    seed_region_next[seed_region_next > upper] = 0

    index = seed_region_next.argmax()
    pos = np.unravel_index(index, seed_region_next.shape)
    seed_next = Pixel(int(pos[0]), int(pos[1]), height)

    if show_seed:
        public.show_seed(slice_next, seed_next, 'tmp/.png', title=f's:{seed_next}')

    if lower < seed_region_next.item(index) < upper:
        logger.debug(f'seed got: {seed_next}, value of seed: {seed_region_next.item(index)}')
        return seed_next, slice_next
    else:
        return None, slice_next


def grow_by_every_slice(seed_first: Pixel, img: np.ndarray, ratio: float = 0.5,
                        verbose: bool = False, min_iter: int = 3
                        ) -> Tuple[np.ndarray, float, float]:
    """
    :param seed_first:
    :param img:
    :param ratio:
    :param verbose:
    :param min_iter:
    :return: mask, mean, std
    """
    seed = seed_first
    slice_ = seed.get_slice(img)
    seed_region_intensity = seed.get_neighborhood_3d_arr(img, half_width=5, half_height=0).mean()

    optimized_threshold, trigger_ratio = get_optimized_threshold(
        slice_, seed, seed_region_intensity, ratio, verbose, min_iter)
    mask_res_ = region_grow(slice_, seed, optimized_threshold).astype(np.float)

    if verbose:
        public.draw_mask_contours(slice_, mask_res_, 'tmp/.png', title=f's: {seed}')

    mask_3d = np.full(img.shape, np.nan)
    seed.set_slice(mask_3d, mask_res_)

    optimized_thresholds = [optimized_threshold]

    def get_mask_mean_std():
        mask_3d_tmp = (mask_3d * img).astype(np.float)
        # logger.debug(f'{mask_3d_tmp.dtype}, {img.dtype}, {mask_3d.dtype}')
        mask_3d_tmp[mask_3d_tmp == 0] = np.nan
        m, s = np.nanmean(mask_3d_tmp), np.nanstd(mask_3d_tmp)
        return float(m), float(s)

    max_index = img.shape[0] - 1
    mask_res = mask_res_
    while seed.height < max_index:
        mean, std = get_mask_mean_std()
        seed, slice_next = get_seed_in_neighbor_slice(seed, img, mask_res, mean, std,
                                                      increase=True, show_seed=verbose)
        if seed is None:
            # 停止层间生长
            break

        seed_region_intensity = seed.get_neighborhood_3d_arr(img, half_width=5, half_height=0).mean()
        optimized_threshold, _ = get_optimized_threshold(slice_next, seed,
                                                         seed_region_intensity, ratio, verbose,
                                                         min_iter)
        # 对于A100 B10 C100 类型, 阈值可能会非常接近10
        # upper = mean + 3 * std
        # lower = mean - 3 * std
        # if not lower < optimized_threshold < upper:
        #     logger.debug(f'range: ({lower}, {upper}), threshold: {optimized_threshold}')
        #     break

        optimized_thresholds.append(optimized_threshold)
        mask_res = region_grow(slice_next, seed, optimized_threshold)
        seed.set_slice(mask_3d, mask_res)

        if verbose:
            public.draw_mask_contours(slice_next, mask_res, 'tmp/.png', title=f's: {seed}')

    seed = seed_first
    mask_res = mask_res_
    while seed.height > 0:
        mean, std = get_mask_mean_std()
        seed, slice_next = get_seed_in_neighbor_slice(seed, img, mask_res, mean, std,
                                                      increase=False, show_seed=verbose)
        if seed is None:
            # 停止层间生长
            break

        seed_region_intensity = seed.get_neighborhood_3d_arr(img, half_width=5, half_height=0).mean()
        optimized_threshold, _ = get_optimized_threshold(slice_next, seed, seed_region_intensity,
                                                         ratio, verbose, min_iter)

        # upper = mean + 3 * std
        # lower = mean - 3 * std
        # if not lower < optimized_threshold < upper:
        #     logger.debug(f'range: ({lower}, {upper}), threshold: {optimized_threshold}')
        #     break

        optimized_thresholds.append(optimized_threshold)
        mask_res = region_grow(slice_next, seed, optimized_threshold)
        seed.set_slice(mask_3d, mask_res)

        thresholds_arr = np.array(optimized_thresholds)
        logger.debug(f'mean of threshold: {thresholds_arr.mean()}, {thresholds_arr}')
        if verbose:
            public.draw_mask_contours(slice_next, mask_res, 'tmp/.png', title=f's: {seed}')

    if verbose:
        plt.show()
    mean, std = get_mask_mean_std()
    mask_3d[np.isnan(mask_3d)] = 0
    mask_3d = mask_3d.astype(np.uint8)
    return mask_3d, mean, std


def test0(seed: Pixel, show_seed=True, preset_ratio=1.8):
    os.makedirs('tmp/', exist_ok=True)
    nii: nib.nifti1.Nifti1Image = nib.load('test.nii.gz')
    img: np.ndarray = nii.get_fdata()
    # 交换图像数据的存储方向
    img = np.transpose(img, [2, 1, 0])
    logger.debug(f'shape of the NIfTI image: {img.shape}, seed: {seed}')

    slice_: np.ndarray = img[seed.height]

    slice_copy = slice_.copy()
    selected_area = slice_[seed.row - 5:seed.row + 5, seed.col - 5:seed.col + 5]
    # 对矩阵进行修改要非常谨慎
    slice_copy[seed.row - 5:seed.row + 5, seed.col - 5:seed.col + 5] = 0
    # logger.debug('selected area: ', selected_area)
    img_show(slice_copy)

    figure_seed_region, axis_seed_region = img_show(slice_)

    # 左上角角点
    left_up = (seed.col - 5, seed.row - 5)
    rect = plt.Rectangle(left_up, 10, 10, edgecolor='red', fill=False)
    axis_seed_region.add_patch(rect)
    figure_seed_region.savefig('tmp/seed_region.png')

    reference_intensity = selected_area.sum() / selected_area.size
    logger.debug(f'reference intensity: {reference_intensity}, sum: {selected_area.sum()}, num: {selected_area.size}')
    optimized_threshold, trigger_ratio = get_optimized_threshold(slice_, seed, reference_intensity, ratio=preset_ratio,
                                                                 verbose=False)

    logger.debug(f'threshold: {optimized_threshold: .1f}, target: 1200, deviation: {optimized_threshold - 1200}')
    # TODO remove the return
    # return

    mask = region_grow(slice_, seed, optimized_threshold)

    figure_seed_segmentation, axis_seed_segmentation = img_show(mask * slice_)
    figure_seed_segmentation.savefig('tmp/seed_mask.png')

    mask_3d = region_grow_3d(img, seed, optimized_threshold)

    mask_3d = np.transpose(mask_3d, [2, 1, 0])
    # P: pixel value
    # RI: reference intensity 平均像素强度
    # OT: optimized threshold
    # PR: preset ratio
    # TR: trigger ratio
    # V: volume
    work_dir = f'output/({seed.height},{seed.row},{seed.col})' \
               f'-P{int(seed.get_pixel_3d(img))}' \
               f'-RI{int(reference_intensity)}' \
               f'-OT{int(optimized_threshold)}' \
               f'-PR{preset_ratio}' \
               f'-TR{trigger_ratio: .1f}' \
               f'-V{mask_3d.sum() // 100}'
    os.makedirs(work_dir, exist_ok=True)

    # TODO 这个操作不好
    nii._dataobj = mask_3d
    nii.update_header()
    nib.nifti1.save(nii, f'{work_dir}/mask.nii.gz')

    os.system(f'rm tmp/tmp* && mv tmp/* "{work_dir}"')

    if show_seed:
        plt.show()


def test1(seed: Pixel):
    os.makedirs('tmp/', exist_ok=True)
    nii: nib.nifti1.Nifti1Image = nib.load('test.nii.gz')
    img: np.ndarray = nii.get_fdata()
    # 交换图像数据的存储方向
    img = np.transpose(img, [2, 1, 0])
    save_nii(img, 'tmp/test1.nii.gz')
    logger.debug(f'shape of the NIfTI image: {img.shape}, seed: {seed}')
    start = time.time()
    mask = region_grow_3d_without_threshold(img, seed)
    time_cost = time.time() - start
    save_nii(mask, 'tmp/test1_mask-seed2-without-realtime-threshold.nii.gz')


def test():
    seeds_ = list()
    seeds_.append(Pixel(220, 350, 11))
    seeds_.append(Pixel(220, 340, 10))
    seeds_.append(Pixel(220, 350, 9))
    seeds_.append(Pixel(220, 350, 8))
    seeds_.append(Pixel(220, 350, 7))
    seeds_.append(Pixel(220, 350, 6))
    for s_ in seeds_:
        test0(s_, False, 1.1)
    # test1(seeds_[2])
    # test1(seeds_[3])


if __name__ == '__main__':
    os.system('rm -rf output/*220,3*')
    test()
