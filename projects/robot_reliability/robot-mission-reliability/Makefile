.PHONY: benchmark experiment test replay

benchmark:
	PYTHONPATH=src python -m robot_lab.cli experiment --output artifacts --docs-output docs

experiment: benchmark

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

replay:
	PYTHONPATH=src python -m robot_lab.cli replay artifacts/runs/severe_slip/v2_seed_0.jsonl