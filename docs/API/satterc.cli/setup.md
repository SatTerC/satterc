# `satterc.cli.setup`

Generate a configuration file for SatTerC.

## Usage

```sh
satterc setup [OPTIONS]
```

## Options

| Flag | Short | Description |
|------|-------|-------------|
| `--models` | `-m` | Space-separated list of built-in models to include (e.g., `splash pmodel`). If not provided, runs interactive selector. |
| `--output` | `-o` | Output path for the generated config file (default: `config.toml`) |
| `--defaults` | `-d` | Use default input/output paths without interactive prompting (requires `--models`) |

## API Documentation

::: satterc.cli.setup
