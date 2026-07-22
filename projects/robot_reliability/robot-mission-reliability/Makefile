.PHONY: experiment test replay

experiment:
	PYTHONPATH=src python -m robot_lab.cli experiment --output artifacts

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

replay:
	PYTHONPATH=src python -m robot_lab.cli replay artifacts/runs/v1.jsonl