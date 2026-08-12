.PHONY: check test example scaffold-example

check:
	uv run python -m compileall -q tileops_foundry_loop scripts tests examples templates
	uv run python scripts/check_pr.py examples/fused-moe/pr-data.json
	@for file in scripts/*.sh; do bash -n "$$file"; done

test:
	uv run python -m unittest discover -s tests -v

example:
	uv run python scripts/render_pr.py examples/fused-moe/pr-data.json \
		--output-dir examples/fused-moe

scaffold-example:
	uv run python scripts/new_round.py --slug example --scope GEMM --operator ExampleOp
