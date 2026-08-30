_: lint typecheck test

# Format and lint the package using ruff, and lint the recipe notebooks using marimo.
lint:
  ruff format
  ruff check --fix
  marimo check recipes/

# Variant of `lint` that doesn't cause any changes to files.
lint-check:
  ruff format --check
  ruff check
  marimo check recipes/

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

# Render a model's DAG to an SVG inlined by its docs page.
graph model:
  uv run python scripts/render_model_graph.py {{model}} docs/models/_graphs/{{model}}.svg

# Re-render every model graph in docs/models/_graphs/.
graph-all:
  just graph splash
  just graph pmodel
  just graph sgam
  just graph rothc

# Build the documentation using Zensical.
# Depends on `graph-all` because the model graphs are generated, not committed:
# rendering one takes well under a second, so there is nothing to gain by
# checking in an SVG that a signature change would silently invalidate.
# `--clean` because zensical's incremental cache does not track the files pulled
# in by `pymdownx.snippets`, so a freshly rendered graph is otherwise ignored.
docs: graph-all
  zensical build --clean

# Export a single recipe notebook to docs/recipes/.
export recipe:
  # Export to Markdown and keep the HTML notebook in the generated assets directory.
  marimo-md-export "recipes/{{recipe}}.py" "docs/recipes/{{recipe}}.md" \
    --keep-html --overflow scroll

# Export all notebooks in recipes/ to docs/recipes/.
export-all:
  just export my_first_pipeline
  just export full_pipeline
  just export soil_moisture
  just export pft_parameters
