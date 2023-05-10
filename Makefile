.PHONY: pre-commit format clean run
export PYTHONPATH := $(PWD)

format:
	poetry run isort . & poetry run black .
	poetry run autoflake -ri --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables .

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

run %:
	poetry run python3 main.py $(filter-out $@,$(MAKECMDGOALS))
%:
	@:
