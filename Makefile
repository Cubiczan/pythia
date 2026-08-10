.PHONY: install check test lint format clean paper-trade live-run backtest

PYTHON ?= python3
PIP ?= pip

# Install packages in dependency order: leaf packages first, executor & forge last.
PACKAGES := \
    packages/delphi-adapter \
    packages/strata \
    packages/analyst-mesh \
    packages/consensus \
    packages/risk \
    packages/observability \
    packages/executor \
    packages/forge

install:
	@echo "==> Installing packages (editable) in dependency order"
	@for pkg in $(PACKAGES); do \
		echo "  -> $$pkg"; \
		(cd $$pkg && $(PIP) install -e . --quiet) || exit 1; \
	done
	@echo "==> Done. Verify with 'make check'."

check:
	@echo "==> Verifying environment"
	@$(PYTHON) -c "import os; assert os.getenv('DELPHI_API_KEY'), 'DELPHI_API_KEY not set'" || true
	@$(PYTHON) -c "import os; assert os.getenv('LLM_API_KEY'), 'LLM_API_KEY not set'" || true
	@$(PYTHON) -c "from pythia_delphi_adapter.client import DelphiClient; print('  adapter import OK')"
	@$(PYTHON) -c "from pythia_analyst_mesh.registry import AnalystRegistry; print('  mesh import OK')"
	@$(PYTHON) -c "from pythia_consensus.fusion import fuse; print('  consensus import OK')"
	@$(PYTHON) -c "from pythia_risk.sizing import size_trade_kelly; print('  risk import OK')"
	@$(PYTHON) -c "from pythia_executor.pipeline import PythiaExecutor; print('  executor import OK')"
	@$(PYTHON) -c "from pythia_forge.backtester import Backtester; print('  forge import OK')"
	@$(PYTHON) -c "from pythia_observability.audit_reader import AuditLogReader; print('  observability import OK')"
	@$(PYTHON) -c "from pythia_strata.enricher import MarketEnricher; print('  strata import OK')"
	@echo "==> All imports OK"

test:
	@for pkg in $(PACKAGES); do \
		echo "==> Testing $$pkg"; \
		(cd $$pkg && $(PYTHON) -m pytest tests/ -q) || exit 1; \
	done

lint:
	@for pkg in $(PACKAGES); do \
		(cd $$pkg && ruff check src/ tests/) || true; \
	done

format:
	@for pkg in $(PACKAGES); do \
		(cd $$pkg && ruff format src/ tests/) || true; \
	done

clean:
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/ .ruff_cache/

paper-trade:
	@$(PYTHON) -m pythia_executor.cli executor delphi paper-trade $(ARGS)

live-run:
	@$(PYTHON) -m pythia_executor.cli executor delphi run $(ARGS)

backtest:
	@$(PYTHON) -m pythia_forge.cli backtest $(ARGS)
