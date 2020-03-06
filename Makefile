PY = python
PIP = pip

install:
        $(PIP) install numpy
        $(PIP) install scipy

all:

				for i in {0.0001..1..0.1} do
					$(PY) "main.py" $(i)
				done

submit-all:

				for i in {0.0001..1..0.1} do
					$(PY) "main.py" $(i)
					kaggle competitions submit -c project-2-cs529 -f output.csv -m "Beta = ${i}"
				done
