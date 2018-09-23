import setuptools

# name = *
# version = *
# author
# author_email
# maintainer
# maintainer_email
# url
# license
# description = *
# long_description
# platforms = 
# classifiers = 
# keywords = 
# packages = *
# py_modules = 
# download_url = 
# package_data = *
# include_package_data = 
# exclude_package_data = 
# data_files = 
# ext_modules = *
# scripts = *
# package_dir = *
# requires = 
# provides = 
# install_requires = *
# entry_points = 
# setup_requires = 
# dependency_links = 



with open("README.md", "r") as fh:
    long_description = fh.read()

print setuptools.find_packages()

setuptools.setup(
    name="demo_project",
    version="0.0.1",
    author="guru4elephant",
    author_email="guru4elephant@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guru4elephant/DL-arsenal.git",
    packages=['moduleA', 'moduleB', 'moduleA.lego'],
    package_data={'moduleA': ['data/dat*.txt']},
    package_dir={'moduleA.lego.x': '/home/dongdaxiang/github_develop/DL-arsenal/test_py'},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
