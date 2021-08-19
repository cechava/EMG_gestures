import sys
import os
from setuptools import setup, find_packages
PACKAGES = find_packages()


VERSION = '0.1.0'

INSTALL_REQ = [
''
]

opts = dict(name = 'limb_position_EMG',
            maintainer='Cesar Echavarria',
            maintainer_email='cechavarria60@gmail.com',
            description='package with functions to analyze EMG data',
            author='Cesar Echavarria',
            author_email='cechavarria60@gmail.com',
            version=VERSION,
            # Installation requirements
    		install_requires= INSTALL_REQ,
            packages=PACKAGES)

if __name__ == '__main__':
    setup(**opts)