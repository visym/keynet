from setuptools import setup, find_packages

d_version = {}
with open("./keynet/version.py") as fp:
    exec(fp.read(), d_version)
version = d_version['VERSION']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='keynet',
    author='Jeffrey Byrne',
    author_email='jeff@visym.com',
    version=version,
    packages=find_packages(),
    description='__________________________________',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/visym/keynet', 
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "vipy",
        "scikit-learn",
        "joblib",
        "tqdm"
    ],
    keywords=['vision', 'learning', 'ML', 'CV'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)
