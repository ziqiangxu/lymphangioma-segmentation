"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def img_show(img, size=(6, 6), output='tmp/tmp.png'):
    figure, axes = plt.subplots(figsize=size)
    axes.imshow(img, cmap='gray')
    figure.savefig(output)
    return figure, axes


def save_img(img, path, size=(6, 6)):
    figure, axes = plt.subplots(figsize=size)
    axes.imshow(img, cmap='gray')
    figure.savefig(path)
    plt.close(figure)


def draw_curve(x_data, y_data, output=None, size=(6, 6), x_label='', y_label=''):
    figure, axes = plt.subplots(figsize=size)
    axes.plot(x_data, y_data)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    if output:
        figure.savefig(output)
        plt.close(figure)
    return figure, axes
