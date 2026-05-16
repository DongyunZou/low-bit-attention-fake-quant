# low-bit-fake-quant

Triton-first helpers for low-bit fake-quant precision experiments.

The initial scope is attention activation quantization for video diffusion
experiments:

- Q/K FP8 e4m3fn with FP32 block scales.
- Q/K MXFP8-style block scaling.
- Per-channel FP8 V quantization.
- K smoothing, Q centering, and optional Q k-means reorder helpers.
- Small-shape P requant probes for validating FP8 probability casting.

See `docs/quant_precision_test_plan.md` for the implementation and evaluation
plan.

## Setup

```bash
uv sync --extra dev --extra bench
uv run pytest
```
