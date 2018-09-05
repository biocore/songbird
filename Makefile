.PHONY: all lint test test-cov install dev clean distclean

all: ;

lint:
	q2lint
	flake8

test: all
	py.test

test-cov: all
	py.test --cov=songbird

install: all
	python setup.py install

dev: all
	pip install -e .

clean: distclean

distclean: ;
