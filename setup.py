#responsible for creating your machine learning model 
# as a package
# and deploy it in pypi for others to use

from setuptools import find_packages,setup
# import src
from typing import List

HYPENEDOT = '-e .'

def get_requirements(filepath:str)->List['str']:
    with open(filepath) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPENEDOT in requirements:
            requirements.remove(HYPENEDOT)
    return requirements            

setup(
name = 'mlproject',
version='0.0.1',
author = 'Abhinav Prabhakar',
author_email= 'abhinav2presidency@gmail.com',
description="A simple end-to-end ML Project",
packages=find_packages(),
install_requires= get_requirements('requirements.txt')
)