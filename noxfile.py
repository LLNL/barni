import shlex

import nox
@nox.session(python="3.6")
def tests(session):


    # install requirements
    session.install('pyyaml>=5.1')
    session.install('numpy>=1.17')
    session.install('scipy>=1.3.0')
    session.install('scikit-learn>=0.20.0')
    session.install('bokeh>=1.4.0')
    session.install('pandas>=0.25')
    session.install('coverage')
    session.install('cython>=0.2')

    # build
    session.run(*shlex.split("python setup.py sdist"))
    session.run(*shlex.split("pip install dist/BARNI-1.0.0.tar.gz"))

    #run coverage
    session.run(*shlex.split('coverage run -m unittest discover -v'))
    session.run(*shlex.split('coverage report --include="barni*" --omit="test*"'))
