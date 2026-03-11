docs:
  zensical build

test:
  pytest

lint:
  ruff format src/
  ruff check src/

viz:
  satterc graph config.toml --pdf
  zathura pipeline.pdf

run:
  satterc run config.toml
