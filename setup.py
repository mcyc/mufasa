from setuptools import setup, find_packages

setup(
    name='mufasa',
    version='1.3.1',
    description='MUlti-component Fitter for Astrophysical Spectral Applications',
    author='Michael Chun-Yuan Chen',
    author_email='mkid.chen@gmail.com',
    url='https://github.com/mcyc/mufasa',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy',
        'matplotlib',
        'scipy',
        'scikit-image',
        'spectral-cube',
        'radio-beam',
        'pvextractor',
        'pandas',
        'reproject',
        'pyspeckit @ git+https://github.com/pyspeckit/pyspeckit@master#egg=pyspeckit',
        'FITS_tools @ git+https://github.com/pyspeckit/FITS_tools@py312#egg=FITS_tools'
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

