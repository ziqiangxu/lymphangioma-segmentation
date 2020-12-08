"""
Author: Daryl.Xu
E-mail: ziqiang_xu@qq.com

可见logging相关介绍 https://cuiqingcai.com/6080.html
"""
import logging


def create_develop_logger():
    logger = logging.getLogger('lymphangioma_segmentation')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        # "%(asctime)s %(name)s %(filename)s %(lineno)s %(levelname)s %(message)s",
        "%(asctime)s.%(msecs)03d %(filename)s %(lineno)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('develop logger has setup!')
    return logger


def _setup_logger():
    logger = logging.getLogger('lymphangioma_segmentation')
    if logger.handlers:
        raise BaseException('_setup_logger()只允许在导入lymphangioma_segmentation这个库的时候'
                            '被调用一次，防止添加多个标准输出handler')
    return create_develop_logger()


def get_logger(name: str):
    """
    有 lymphangioma_segmentation. 前缀则可继承当前的配置
    :param name:
    :return:
    """
    return logging.getLogger(name)


def create_product_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.INFO)
    formatter = logging.Formatter(
        # "%(asctime)s %(name)s %(filename)s %(lineno)s %(levelname)s %(message)s",
        "%(asctime)s.%(msecs)03d %(filename)s %(lineno)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    # TODO file handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger_.addHandler(handler)
    return logger_
