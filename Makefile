train:
	python -m src.run --input data/processed/features.csv
test:
	pytest tests/ -q
