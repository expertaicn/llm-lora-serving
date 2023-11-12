from setuptools import setup, find_packages
import regex as re
import os

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="llmserver",
    version=find_version(get_path("llmserver", "__init__.py")),
    packages=find_packages(),
    install_requires=[
        # 你的程序依赖的其他库列表，例如：
        # 'numpy',
        # 'requests',
    ],
    entry_points={
        "console_scripts": [
            # 如果你的程序是一个命令行工具，你可以在这里指定入口点，例如：
            # '命令行工具名=模块名:函数名',
        ],
    },
    # 其他的元数据，如作者，授权信息等：
    author="你的名字",
    author_email="你的邮件地址",
    description="程序的简短描述",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # 如果你的README是Markdown格式的
    url="项目的网址或者源代码仓库的网址",
)
