# Default recipe to display available commands.
_:
  @just --list

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

# Run the full test suite.
test: lint
  pytest

# Build the documentation using Zensical.
docs:
  zensical build

# Export a single example notebook to docs/Examples/.
export example:
  # Export to Markdown file
  marimo-md-export "examples/{{example}}.py" "docs/Examples/{{example}}.md" \
    --html-output docs/Examples/{{example}}-notebook.html

# Export all notebooks in examples/ to docs/Examples/.
export-all:
  #just export 00-getting-started
  #just export 00-getting-started-csv
  #just export 01-demo
  just export 02-soil-moisture
  just export 02-soil-moisture-csv
  just export 03-pft-parameters
  just export 03-pft-parameters-csv
  
