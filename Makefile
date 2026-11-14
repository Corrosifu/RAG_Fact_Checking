.PHONY: install test lint format demo serve ui e2e
install:  ## dev install
	pip install -e ".[dev,ui]"
test:  ## run tests
	pytest --cov --cov-report=term-missing
lint:
	ruff check . && ruff format --check .
format:
	ruff check --fix . && ruff format .
demo:  ## offline demo on the bundled corpus
	ragfc sample && ragfc index && ragfc ask "How does quantization reduce model size?"
serve:
	ragfc serve
ui:
	streamlit run ui/streamlit_app.py
e2e:  ## end-to-end regression gate (lightweight components, golden question set)
	ragfc e2e --baseline tests/e2e/baseline.json
