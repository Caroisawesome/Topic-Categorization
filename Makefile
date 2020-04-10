PY = python
PIP = pip

install:
				$(PIP) install numpy
				$(PIP) install scipy

all:
				$(PY) util.py

