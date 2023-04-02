from setuptools import setup, find_packages

setup( 
    name = 'oidn',
    version = '0.1',
    author = 'HfCloud',
    author_email = 'sxysxygm@gmail.com',
    description = 'A simple python binding for Intel OIDN',
    packages=find_packages(),
    install_requirs=[
        'numpy >= 1.12.0',
        'pillow'
    ]
)