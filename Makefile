# risk-model
PY ?= python3
RUN = PYTHONPATH=src $(PY)

.PHONY: help install test results figures paper prototype all clean

help:
	@echo "make install | test | results | figures | paper | prototype | all | clean"

install:               ## install the package and test dependencies
	$(PY) -m pip install -e ".[dev]"

test:                  ## unit tests, reconciliation and prototype parity
	$(RUN) -m pytest -q tests -m "not slow"

results:               ## retrain the pipeline on synthetic data (about 30 s) and write results/results.json
	$(RUN) -m risk_model.pipeline

figures: results       ## regenerate paper/fig/*.pdf from results/results.json
	$(RUN) -m risk_model.figures

paper: figures         ## build paper/paper.tex and paper/paper.pdf (needs a LaTeX install)
	$(PY) paper/build_paper.py
	cd paper && pdflatex -interaction=nonstopmode paper.tex >/dev/null && pdflatex -interaction=nonstopmode paper.tex >/dev/null

prototype: results     ## build prototype/index.html (single-file HTML demo)
	$(PY) prototype/build_prototype.py

all: test results figures paper prototype

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc
