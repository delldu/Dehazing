"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="image_dehaze",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="image/video dehaze package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/Dehazing.git",
    packages=["image_dehaze"],
    package_data={"image_dehaze": ["models/image_dehaze.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.9.0",
        "torchvision >= 0.10.0",
        "Pillow >= 7.2.0",
        # "numpy >= 1.19.5",
        "einops >= 0.3.0",
        # "redos >= 1.0.0",
        "todos >= 1.0.0",
    ],
)
