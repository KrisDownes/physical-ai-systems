.PHONY: install benchmark experiment test replay

install:
	python -m pip install -e .

benchmark:
	python -m robot_lab.cli experiment --output artifacts --docs-output docs

experiment: benchmark

test:
	python -m unittest discover -s tests -v

replay:
	python -m robot_lab.cli replay artifacts/runs/severe_slip/v2_seed_0.jsonl