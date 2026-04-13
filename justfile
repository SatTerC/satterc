# Default recipe to display available commands.
_:
  @just --list

# Format and lint the package using ruff.
lint:
  ruff format src/
  ruff check src/

# Run the full test suite.
test:
  pytest
  # TODO: integration test

# Build the documentation using Zensical.
docs:
  zensical build

# Export a single example notebook to the docs.
_export example:
  # Export as Markdown file (won't have rendered outputs, sadly)
  marimo export md "examples/{{example}}/notebook.py" --output "docs/Examples/{{example}}.md" --force

  # Export as static HTML
  marimo export html "examples/{{example}}/notebook.py" --output "docs/Examples/{{example}}-notebook.html" --force

  # TODO: both of these commands could be run using --sandbox

# Export all notebooks in examples/ to Markdown+HTML in docs/Examples/.
export-examples:
  just _export 01-demo
