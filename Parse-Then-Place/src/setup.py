from setuptools import find_packages, setup

setup(
    name='text2layout',
    version='0.0.1',
    description='text2layout',
    packages=find_packages(exclude=["test_*.py"]),
    install_requires=[
    ],
)