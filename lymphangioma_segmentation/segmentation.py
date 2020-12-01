"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import math
import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from lymphangioma_segmentation.image import Pixel
from lymphangioma_segmentation.public import img_show, draw_curve, save_img, save_nii

# def pixel_like(pixel_value: float, reference_intensity: float, threshold: float):
#     if pixel_value > reference_intensity or reference_intensity - pixel_value < threshold:
#         return True
#     else:
#         return False


def get_optimized_threshold(img: np.ndarray, seed: Pixel, reference_intensity: float,
                            ratio: float, verbose=False) -> float:
    if verbose:
        print(f'seed_value: {seed.get_pixel(img)}, max_value: {img.max()}')

    descent_rate = 0.85

    threshold = reference_intensity * descent_rate
    thresholds = [threshold,]

    # 这个循环可以并行操作
    # 计算公式 (area3 - area2) / (area2 - area1) 大于2左右的一个经验值
    mask = region_grow(img, seed, threshold)
    area1 = mask.sum()

    reference_intensity = (mask * img).sum() / mask.sum()
    threshold_ = reference_intensity * math.pow(descent_rate, 2)
    if threshold > threshold_:
        threshold = threshold_
    else:
        print('descent according to the previous threshold')
        threshold *= descent_rate

    thresholds.append(threshold)
    mask = region_grow(img, seed, thresholds[1])
    area2 = mask.sum()

    reference_intensity = (mask * img).sum() / mask.sum()
    threshold_ = reference_intensity * math.pow(descent_rate, 3)
    if threshold > threshold_:
        threshold = threshold_
    else:
        print('descent according to the previous threshold')
        threshold *= descent_rate

    thresholds.append(threshold)
    mask = region_grow(img, seed, thresholds[2])
    area3 = mask.sum()
    areas = [area1, area2, area3]

    ratios = []
    iteration = 3
    min_iter = 4

    def draw_log_pic():
        x = thresholds[:len(areas)]
        figure, axes = draw_curve(x, areas, x_label='threshold', y_label='area')
        axes.axvline(x=optimized_threshold, linestyle='-', color='red')
        figure.savefig('tmp/area_threshold_curve.png')
        plt.close(figure)

        area_s = np.array(areas[:-1])
        area_g = np.array(areas[1:])
        area_delta = area_g - area_s
        index = np.linspace(1, area_delta.size, num=area_delta.size)
        draw_curve(index, area_delta, 'tmp/area_delta.png', y_label='area_delta')

        index = [i for i in range(len(ratios))]
        draw_curve(index, ratios, 'tmp/ratios_curve.png', x_label='iter', y_label='ratios')
        draw_curve(index[min_iter:], ratios[min_iter:], 'tmp/ratios_curve_min.png', y_label='ratios')

        index = [i for i in range(len(thresholds))]
        draw_curve(index, thresholds, 'tmp/thresholds_curve.png', x_label='iter', y_label='threshold')

    while True:
        # ratio_ = None
        if area2 != area1:
            ratio_ = (area3 - area2) / (area2 - area1)
            ratios.append(ratio_)
            if verbose:
                print(f'ratio: {ratio_: .2f}, threshold: {threshold: .1f}, area3: {area3}, area2: {area2}, area1: {area1}')
            if ratio_ > ratio and iteration > min_iter:
            # if abs(threshold - 1100) < 100:
                save_img(mask * img, 'tmp/critical_mask.png')
                optimized_index = len(areas) - 2
                optimized_threshold = thresholds[optimized_index]
                print(f'The optimized threshold: {optimized_threshold}')
                # show the optimized segmentation
                #             img_show(mask)
                draw_log_pic()
                return optimized_threshold, ratio_
        else:
            ratios.append(0)
        # 如果面积相等，需要打破这个状态，避免下将停止

        area1 = area2
        area2 = area3

        iteration += 1

        # average_intensity = (mask * img).sum() / mask.sum()
        # threshold_ = average_intensity * math.pow(descent_rate, iteration)
        # if threshold > threshold_:
        #     threshold = threshold_
        # else:
        #     print('descent according to the previous threshold')
        #     threshold *= descent_rate
        threshold *= descent_rate

        mask = region_grow(img, seed, threshold)
        area3 = mask.sum()
        areas.append(area3)

        thresholds.append(threshold)

        if iteration > 300:
            break

    ratios_valid = ratios[min_iter:]
    ratio_ = max(ratios_valid)
    index = ratios_valid.index(ratio_)
    optimized_threshold = thresholds[min_iter + index - 1]
    draw_log_pic()
    plt.plot(ratios)
    plt.show()
    return optimized_threshold, ratio_
#     if mask.sum() > area_limit:
#         break
#     print(f'threshold: {threshold}, area: {mask.sum()}')


def region_grow(img: np.ndarray, seed: Pixel, threshold: float) -> np.ndarray:
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
    print(f'seeds_mean: {seeds_mean}, seeds_std: {seeds_std}')

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
        print(f'pixels number: {mask.sum()}')
        for pixel in neighborhoods:
            flag = pixel.get_pixel_3d(mask)
            pixel_value = pixel.get_pixel_3d(img)
            if flag == 0 and compare_pixel(pixel_value):
                seeds.append(pixel)
                pixel.mark(mask)
    return mask


def test0(seed: Pixel, show_seed=True, preset_ratio=1.8):
    os.makedirs('tmp/', exist_ok=True)
    nii: nib.nifti1.Nifti1Image = nib.load('test.nii.gz')
    img: np.ndarray = nii.get_fdata()
    # 交换图像数据的存储方向
    img = np.transpose(img, [2, 1, 0])
    print(f'shape of the NIfTI image: {img.shape}, seed: {seed}')

    slice_: np.ndarray = img[seed.height]

    slice_copy = slice_.copy()
    selected_area = slice_[seed.row - 5:seed.row + 5, seed.col - 5:seed.col + 5]
    # 对矩阵进行修改要非常谨慎
    slice_copy[seed.row - 5:seed.row + 5, seed.col - 5:seed.col + 5] = 0
    # print('selected area: ', selected_area)
    img_show(slice_copy)

    figure_seed_region, axis_seed_region = img_show(slice_)

    # 左上角角点
    left_up = (seed.col - 5, seed.row - 5)
    rect = plt.Rectangle(left_up, 10, 10, edgecolor='red', fill=False)
    axis_seed_region.add_patch(rect)
    figure_seed_region.savefig('tmp/seed_region.png')

    reference_intensity = selected_area.sum() / selected_area.size
    print(f'reference intensity: {reference_intensity}, sum: {selected_area.sum()}, num: {selected_area.size}')
    optimized_threshold, trigger_ratio = get_optimized_threshold(slice_, seed, reference_intensity, ratio=preset_ratio,
                                                                 verbose=False)

    print(f'threshold: {optimized_threshold: .1f}, target: 1200, deviation: {optimized_threshold - 1200}')
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
    print(f'shape of the NIfTI image: {img.shape}, seed: {seed}')
    start = time.time()
    mask = region_grow_3d_without_threshold(img, seed)
    time_cost = time.time() - start
    print(f'time cost: {time_cost}')
    save_nii(mask, 'tmp/test1_mask-seed2-without-realtime-threshold.nii.gz')


def test():
    seeds_ = list()
    seeds_.append(Pixel(220, 350, 11))
    seeds_.append(Pixel(220, 340, 10))
    seeds_.append(Pixel(220, 350, 9))
    seeds_.append(Pixel(220, 350, 8))
    seeds_.append(Pixel(220, 350, 7))
    seeds_.append(Pixel(220, 350, 6))
    # for s_ in seeds_:
    #     test0(s_, False, 1.1)
    # test1(seeds_[2])
    test1(seeds_[3])


if __name__ == '__main__':
    os.system('rm -rf output/*220,3*')
    test()
