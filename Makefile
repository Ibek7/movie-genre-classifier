.PHONY: install test test-fast lint clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

test-fast:
	pytest -q tests/test_models.py tests/test_predict.py

lint:
	python -m compileall -q src tests

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
