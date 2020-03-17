Documentation
=============
BARNI is documented using Numpy style `docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_ and compiled using `Sphinx <http://www.sphinx-doc.org/en/stable/>`_. Documentation lives under the :file:`docs` directory and can be comiled into HTML by invoking the Makefile:

.. code:: bash

    make html

This creates the documentation under :file:`docs/_build/html/index.html`.

Developing
==========

Please feel free to contribute any tool or module addition you may deem useful. Remember, any code will be read more times than it is written so readability counts. Make sure to supplement your code with adaquate examples to make it user-friendly. Any code you write should be self-explenatory and not require additional interaction with future users. 

We generally follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ styleguide, with a few exceptions (like indentation spacing). The important things to keep in mind are:

* Use 4 spaces for indentation
* Limit lines to a maximum of 80 characters
* Use blank lines sparingly
* Avoid trailing white space
* Use inline comments sparingly
* Naming conventions to follow:
  - Module namse should be lowercase
  - Class names use the CapWords (aka CamelCase) convention
  - Function names are all lowercase, with underscores if necessary
  - Variables should be lowercase except module level constants
 