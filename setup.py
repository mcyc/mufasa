from setuptools import setup, find_packages

pkg_vars  = {}

with open("mufasa/_version.py") as fp:
    exec(fp.read(), pkg_vars)

setup(
    name='mufasa',
    version=pkg_vars['__version__'],
    description='MUlti-component Fitter for Astrophysical Spectral Applications',
    author='Michael Chun-Yuan Chen',
    author_email='mkid.chen@gmail.com',
    url='https://github.com/mcyc/mufasa',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'astropy',
        'matplotlib',
        'scipy>=1.7.3',
        'scikit-image>=0.17.2',
        'spectral-cube>=0.6.0',
        'radio-beam',
        'pvextractor',
        'pandas',
        'plotly',
        'reproject>=0.7.1',
        'pyspeckit @ git+https://github.com/pyspeckit/pyspeckit@master#egg=pyspeckit',
        'FITS_tools @ git+https://github.com/keflavich/FITS_tools@master#egg=FITS_tools'
    ],

    classifiers=[
        'Development Status :: 5 - Stable',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

