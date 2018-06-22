from setuptools import find_packages, setup

DESC = 'A python library for manipulating sequential and-inverter gates.'

setup(
    name='py-aiger',
    version='0.4.1',
    description=DESC,
    url='http://github.com/mvcisback/py-aiger',
    author='Marcell Vazquez-Chanlatte',
    author_email='marcell.vc@eecs.berkeley.edu',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aigseqcompose = aiger.utils:parse_and_compose',
            'aigcount = aiger.utils:parse_and_count',
            'aigparcompose = aiger.utils:parse_and_parcompose',
        ],
    },
    install_requires=[
        'bidict',
        'click',
        'funcy',
        'lenses',
        'parsimonious',
        'dd',
        'toposort',
        'lens',
    ],
    packages=find_packages(),
)
