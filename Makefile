.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell compgen -c python | sort -V | uniq | grep -E '^python[0-9]+\.[0-9]+$$' | tail -n 1 | cut -c7-)

# name
.butterfly:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .butterfly
	. .butterfly/bin/activate; .butterfly/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .butterfly/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

butterfly: .butterfly

clean: .butterfly
	rm -rf .butterfly
