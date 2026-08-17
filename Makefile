.PHONY: all test bench figures clean

PY := python3

all: test bench figures

## Run the test suite
test:
	$(PY) -m pytest

## Measure every algorithm -> reports/*.csv
bench:
	$(PY) scripts/run_benchmarks.py

## Render the figures -> reports/figures/*.png
figures:
	$(PY) scripts/make_figures.py

clean:
	rm -rf reports/*.csv reports/*.json reports/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
