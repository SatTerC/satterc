_: lint typecheck test

# Format and lint the package using ruff, and lint the examples using marimo.
lint:
  ruff format
  ruff check --fix
  marimo check examples/

# Variant of `lint` that doesn't cause any changes to files.
lint-check:
  ruff format --check
  ruff check
  marimo check examples/

# Run static type checker.
typecheck:
  pyright

# Run the full test suite.
test:
  pytest --verbose # --log-cli-level=INFO

# Run tests with coverage report.
test-cov:
  pytest --cov=satterc --cov-report=term-missing --cov-fail-under=90

# Re-record the pyrealm unit anchor in tests/data/pyrealm_golden.json.
# Only after reviewing *why* pyrealm's output changed. If a unit convention
# moved, correct the annotation rather than the recorded numbers.
regen-pyrealm-golden:
  uv run python scripts/regen_pyrealm_golden.py

# Build the documentation using Zensical.
docs:
  zensical build

# Export a single example notebook to docs/Examples/.
export example:
  # Export to Markdown file
  marimo-md-export "examples/{{example}}.py" "docs/examples/{{example}}.md" \
    --html-output docs/examples/{{example}}-notebook.html --overflow scroll

# Export all notebooks in examples/ to docs/Examples/.
export-all:
  just export my_first_pipeline
  just export soil_moisture
  just export pft_parameters
  just export full_pipeline
