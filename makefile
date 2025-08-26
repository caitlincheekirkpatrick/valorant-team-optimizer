PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install features train clean

venv:
	python -m venv .venv

install: venv
	$(PIP) install -r requirements.txt

features:
	$(PYTHON) src/features/make_features.py --input data/raw/sample_matches.csv --out data/processed/features.csv

train:
	$(PYTHON) src/models/train.py --input data/processed/features.csv --artifacts_dir artifacts

clean:
	rm -rf artifacts/* data/processed/*
