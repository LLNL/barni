BARNI - Benchmark Algorithm for RadioNuclide Identification
========================================================

BARNI is a software for radionuclide identification from gamma-ray spectra. 

It uses a machine learning approach to train for a variaity of spectroscopic gamma-ray radiation detectors.

Introduction
------------
This README file is meant as a simple overview of the BARNI repository. The complete BARNI documentation is maintained using [Sphinx](http://www.sphinx-doc.org/en/stable/).
The documentation resides in the `doc` folder.

Directory Structure
-------------------
The following is an overview of the top directory structure folders:

* `barni`: Code of the BARNI python package. 
* `doc`: The Sphinx documentation of the BARNI package. 
* `examples`: Input and configuration files for running BARNI identification and training routines
* `test`: Unit tests for the code found in the `barni` folder.

In addition, there are various files on the top directory:

* `barni_cli.py`: BARNI command line interface module.
* `barni_cli.spec`: PyInstaller configuration file.
* `pyinstall.py`: PyInstaller build script. 
* `barni.yml`: Anaconda environment file.
* `nose2.cfg`: Nose2 (unit test) configuration file. 
* `setup.py`: BARNI package installation script. 
* `LICENSE`: The liscence description. 



Required Libraries
------------------
* Python 3.7+
* Numpy 1.17+
* SciKit-Learn 0.20+
* Bokeh 1.4+
* Pandas 0.25+


Generating documentation with Sphinx
------------------------------------
RASE documentation is maintained using [Sphinx](http://www.sphinx-doc.org/en/stable/).
The documentation resides in the `doc` folder.

Install Sphinx from PyPi using
`$ pip install Sphinx`

<!-- For referencing figures by number it is required to install the numfig extension for Sphinx. -->
<!-- Installation is performed with the following steps: -->
<!-- 1. Download and untar the file at this [link](https://sourceforge.net/projects/numfig/files/Releases/sphinx_numfig-r13.tgz/download) -->
<!-- 1. Run `2to3 -w setup.py` -->
<!-- 1. Run `python setup.py install` -->

To update the documentation:
1. `cd $BARNI\doc`, where `$BARNI` is BARNI base folder where rase.pyw lives
1. `make html` to generate html docs
1. `make latexpdf` to generate latex docs and immediately compile the pdf

The documentation is generated in the `doc\_build\` folder


Authors
-------

- Mateusz Monterial, LLNL
- Karl Nelson, LLNL

License
-------

BARNI is released under an MIT license. For more details see the [LICENSE](/LICENSE) file.

SPDX-License-Identifier: MIT

LLNL-CODE-805904
