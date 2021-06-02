SHELL=/bin/bash
LINT_PATHS=gym_fishing/ tests/ setup.py

pytest:
	./scripts/run_tests.sh

type:
	pytype -j auto .

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 79 ${LINT_PATHS}

check-codestyle:
	# Sort imports -- nope, isort and black disagree about env/__init__.py
#	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 79 ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	rm -rf *test.png
	cd docs && make clean

# PyPi package release
release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload dist/*

# Test PyPi package release
test-release:
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

install:
	python -m pip install --upgrade pip
	pip install .[tests,docs]


.PHONY: clean spelling doc lint format check-codestyle commit-checks
