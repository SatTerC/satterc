docs:
  zensical build

test:
  pytest

lint:
  ruff format src/
  ruff check src/

viz:
  satterc graph config.toml
  eog pipeline.png

