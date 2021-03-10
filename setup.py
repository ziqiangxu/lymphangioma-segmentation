import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="lymphangioma-segmentation",
    version="0.0.1",
    author="Daryl.Xu",
    author_email="",
    description="segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ziqiangxu/lymphangioma-segmentation.git",
    packages=setuptools.find_packages(),
    install_requires=['pydicom', 'matplotlib', 'numpy', 'nibabel'],
    entry_points={
    },
    classifiers=(
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ),
)
