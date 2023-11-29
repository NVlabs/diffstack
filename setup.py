from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))


# remove images from README

setup(
    name="diffstack",
    packages=[
        package for package in find_packages() if package.startswith("diffstack")
    ],
    install_requires=[
        "wandb",
        "pytorch-lightning",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="diffstack",
    author="NVIDIA AV Research",
    author_email="",
    version="0.0.1",
    long_description="",
    long_description_content_type="text/markdown",
)
