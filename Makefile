.PHONY: install test lint format ci dgp llm-benchmark compare clean

install:
	uv sync --group dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

ci: lint test

dgp:
	uv run python scripts/run_dgp.py --runs 10 --seed 42 --output dgp_results_v3

llm-benchmark:
	uv run python scripts/run_llm_benchmark.py --style numeric --runs 5 --seed 42 --output llm_numeric_v3

compare:
	uv run python scripts/compare_benchmark.py --dgp dgp_results_v3/dgp_ground_truth.json --llm llm_numeric_v3/llm_numeric_results.json --output comparison_numeric_v3

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf dist/ *.egg-info/ build/
