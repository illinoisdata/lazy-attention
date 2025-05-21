import io
import os
from setuptools import Extension, find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("pod", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="promptcache",
    version="0.0.0",
    description="A copy of PromptCache baseline",
    url="https://github.com/haochengxia/DynamicRAG/tree/main/drag/promptcache",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)
