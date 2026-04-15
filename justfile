# Default recipe to display available commands.
_:
  @just --list

# Format and lint the package using ruff, and lint the examples using marimo.
lint:
  ruff format
  ruff check
  marimo check examples/

# Run the full test suite.
test: lint
  pytest
  # TODO: integration test

# Build the documentation using Zensical.
docs:
  zensical build

# Export a single example notebook to docs/Examples/.
export example:
  # Export to Markdown file
  marimo export md "examples/{{example}}.py" --output "docs/Examples/{{example}}.md" --no-sandbox --force
  # Export to static HTML
  marimo export html "examples/{{example}}.py" --output "docs/Examples/{{example}}-notebook.html" --no-sandbox --force

# Export all notebooks in examples/ to docs/Examples/.
export-all:
  just export 00-getting-started
  just export 01-demo
  just export 02-soil-moisture
  just export 03-pft-parameters
  
