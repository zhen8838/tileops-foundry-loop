.PHONY: check

check:
	python -m compileall -q scripts
	for file in scripts/*.sh container/*.sh integrations/foreman/*.sh; do bash -n "$$file"; done
