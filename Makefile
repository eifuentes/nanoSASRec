.PHONY: clean clean-data clean-runs clean-all fresh

# just nuke downloaded datasets (keep trained models)
clean-data:
	rm -rf data/

# nuke training artifacts (keep downloaded datasets)
clean-runs:
	rm -rf checkpoints/ runs/ *.log

# remove python cache only (safe, never destructive)
clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# nuke everything - data, runs, caches
clean-all:
	rm -rf data/ checkpoints/ runs/ *.log __pycache__/ **/__pycache__/

# full reset and venv sync
fresh: clean-all
	uv sync