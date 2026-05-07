# `satterc.cli.data_gen`

Generate synthetic input data for testing.

## Usage

```sh
satterc data-gen generate <CONFIG_FILE> [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `CONFIG_FILE` | Path to a TOML configuration file |

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--grid`, `-g` | Grid dimensions as `n_lat,n_lon` (default: `1,1`) |
| `--duration`, `-d` | Time duration: `2y`, `6m`, `30d` (default: `2y`) |
| `--seed`, `-s` | Random seed for reproducibility (default: `42`) |

## API Documentation

::: satterc.cli.data_gen
