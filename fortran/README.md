# Fortran models via f2py

An illustrative wrapping of [ACM-GPP-ET](https://github.com/GCEL/ACM_GPP_ET) as a satterc node.

```console
just build-acm          # needs gfortran; fetches upstream at a pinned commit
pytest tests/test_acm_gpp_et_seam.py
```

The tests skip, and `[acm_gpp_et]` fails with a pointer to `just build-acm`, until the extension is built.

## How the pieces fit

| Layer | File | Role |
|---|---|---|
| Upstream Fortran | `build/acm_gpp_et/upstream/src/ACM_GPP_ET.f90` | `CARBON_MODEL`, fetched at build time, never edited |
| Shim | `fortran/acm_gpp_et/acm_gpp_et_f2py.f90` | One-pixel subroutine with an f2py-friendly signature |
| Extension | `src/satterc/models/_acm_gpp_et.*.so` | Built by f2py, gitignored |
| Stub | `src/satterc/models/_acm_gpp_et.pyi` | Types for pyright, since f2py emits none |
| Node module | `src/satterc/models/acm_gpp_et.py` | Same shape as `rothc.py`: kernel, `apply_ufunc` seam, config node, `@extract_fields` model node |

The Hamilton node never sees Fortran.
`acm_gpp_et` unpacks `DataArray`s, `_acm_gpp_et` maps over `pixel` with `apply_ufunc`, and `_acm_gpp_et_1px` packs one pixel's drivers into the `met` array and calls `acm_pixel`, which looks like any other Python function:

```python
fluxes, soil_water, wswp, status = acm_pixel(met, pars, lat, clay, sand)
```

## What f2py does

f2py reads a Fortran signature and generates a C extension module that converts NumPy arrays to Fortran arrays and back.

1. **Scan.** `f2py shim.f90 -m _acm_gpp_et -h _acm_gpp_et.pyf` writes a signature file: each argument's type, shape and `intent`.
2. **Compile.** `f2py -c _acm_gpp_et.pyf shim.f90 upstream.f90` generates the C wrapper and compiles everything into one `.so`. On Python ≥ 3.12 the build backend is meson, so meson and ninja must be importable; `build-acm` supplies them with `uv run --with`.

What the signature controls:

- `intent(in)` arrays are inputs; `intent(out)` arrays are allocated by the wrapper and returned, as a tuple when there are several.
- Size arguments that appear in array shapes become optional and are inferred, so `nodays` comes from `met.shape[1]`.
- Fortran is column-major. f2py copies a C-ordered input to Fortran order silently, and keeps the indices meaning the same thing: `met[i, j]` in NumPy is `met(i+1, j+1)` in Fortran. The kernel builds `met` with `np.asfortranarray` so that copy happens once and visibly.
- Outputs come back Fortran-ordered, so `fluxes[:, k]` is a contiguous column.

## Things learned getting ACM-GPP-ET to build and run

- **Upstream is shaped for R, not Python.** `racmgppet` takes pre-allocated output arrays as arguments and flattens everything for `.Fortran`. Wrapping it directly works but gives a poor Python signature, hence the shim. The shim calls `CARBON_MODEL` itself and reproduces `racmgppet`'s setup (soil fractions, `deltat`).
- **f2py cannot scan the upstream source.** It contains a non-ASCII character in a comment, which crashes f2py's scanner. Scanning only the shim and handing upstream to the compiler sidesteps this, and keeps f2py from trying to wrap upstream's module variables.
- **Module state outlives the call.** Fortran module variables live as long as the loaded `.so`, which is the Python process. `CARBON_MODEL` resets most of its state per call, and `test_repeat_call_is_identical` checks that, but it allocates three arrays sized by the first call's `nodays` and never resizes them. A later call with a different series length would read stale values or run off the end of the array. The shim refuses such calls with `status = 1`. A small upstream patch (reallocate when the size changes) would remove the restriction.
- **Not thread-safe, by construction.** Shared module state means two pixels must never run at once in one process. f2py wrappers hold the GIL for the whole call unless the signature says `threadsafe`, so today this is safe and threads give no speed-up. Parallelism has to come from processes, which suits conduit's blocking.
- **No `stop` across the boundary.** A Fortran `stop` exits the whole Python process. The shim returns a `status` code instead, and the kernel raises a `RuntimeError`. Upstream's only other side effect is a `print` on a soil-water imbalance.
- **Results match upstream's R interface.** On `inputs/ACM_GPP_ET_forest_drivers.csv`, the f2py build agrees with `racmgppet` called from R to within 1e-8 on GPP, transpiration and soil water when R's library is built with upstream's recipe (gfortran's default `-O0`), and to within 1e-13 when it is built with `-O3`, as meson does. The difference is optimisation, not the wrapping.
- **Licence.** The upstream repository has no licence, so its source is fetched at build time rather than committed here. Redistributing a compiled wheel would need the authors' permission either way.

## Decisions and known limitations

- **Output names.** `canopy_gpp_daily`, `surface_runoff_daily` etc. avoid colliding with `gpp_daily` (an input in `recipes/config.toml`) and SPLASH's `runoff_daily`. Kept as they are for now.
- **Soil layers.** ACM-GPP-ET has four soil layers, each with its own clay and sand content. `clay_content` and `sand_content` are each either `(pixel,)`, applied to every layer, or `(pixel, soil_layer)` with `soil_layer` of length 1 or 4, top layer first. The two are broadcast independently, so a layered clay profile can sit alongside a single sand value. Upstream's R scripts pass topsoil values for layers 1–2 and subsoil values for 3–4.
- **Foliar nitrogen is one value for every pixel.** It is a config setting because it is an entry of the model's `pars` vector, but it varies with vegetation and site fertility, and GPP scales directly with it. A spatially varying run needs it to become a per-pixel input; not addressed yet.
- **New inputs.** `temperature_min_daily`, `temperature_max_daily`, `shortwave_radiation_daily`, `fine_root_carbon_daily` and `sand_content` are new to satterc. `fine_root_carbon_daily` could come from SGAM's root pool; upstream's test uses `80 × LAI`.
- **CI.** The extension is not built in CI, so the tests skip there. `gfortran` is available on GitHub's Ubuntu runners, so adding `just build-acm` to the workflow is cheap.

## Shipping it properly

The build here is for local experiments: the `.so` is dropped into the source tree of an editable install.
satterc builds with `uv_build`, which cannot compile Fortran, so a real version should follow `rothc-py` and `sgam`: a separate package (say `acm-gpp-et-py`) built with `meson-python`, which supports f2py directly and produces wheels per platform.
satterc then depends on it and keeps only `acm_gpp_et.py`.
