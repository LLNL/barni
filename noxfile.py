import shlex
import nox

nox.options.sessions = ["report"]

@nox.session(python=["3.5", "3.6", "3.7", "3.8"])
def tests(session):


    # install requirements
    session.install('-r', 'requirements.txt')
    # build
    session.run(*shlex.split("python setup.py sdist"))
    session.run(*shlex.split("pip install dist/BARNI-1.0.0.tar.gz"))
    #run tests
    session.run(*shlex.split("python setup.py test"))
    
@nox.session    
def coverage(session):
    # install requirements
    session.install('-r', 'requirements.txt')

    # build
    session.run(*shlex.split("python setup.py sdist"))
    session.run(*shlex.split("pip install dist/BARNI-1.0.0.tar.gz"))

    #run coverage
    session.run(*shlex.split('coverage run -m unittest discover -v'))
    session.run(*shlex.split('coverage xml --include="barni*" --omit="test*"'))

@nox.session    
def report(session):
    # install requirements
    session.install('-r', 'requirements.txt')

    # build
    session.run(*shlex.split("python setup.py sdist"))
    session.run(*shlex.split("pip install dist/BARNI-1.0.0.tar.gz"))

    #run coverage
    session.run(*shlex.split('coverage run -m unittest discover -v'))
    session.run(*shlex.split('coverage report --include="barni*" --omit="test*"'))
