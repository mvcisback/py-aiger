from setuptools import find_packages, setup

setup(
    name='py-aiger',
    version='0.1',
    description='TODO',
    url='http://github.com/mvcisback/py-aiger',
    author='Marcell Vazquez-Chanlatte',
    author_email='marcell.vc@eecs.berkeley.edu',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aigcompose = aiger.utils:parse_and_compose',
            'aigcount = aiger.utils:parse_and_count',
        ],
    },
    install_requires=[
        'bidict',
        'click',
        'funcy',
        'lenses',
        'parsimonious',
        'dd',
    ],
    packages=find_packages(),
)
