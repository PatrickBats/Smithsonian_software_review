# Tests

```bash
python -m unittest discover -s tests -v
```

Checks cover category/split preservation, configuration path origins and overrides,
invalid boxes, crop padding/color, focal fusion, one-to-one matching, ignore-region
loss gradients, and checkpoint-head initialization. Structure checks cover guarded
entrypoints and cleared notebook outputs. Model-dependent tests skip explicitly if
requirements are unavailable.

`smoke_training.py` checks classifier forward/backward arithmetic and gradients
on synthetic tensors. Saved weights remain unchanged. Fresh-install and real
notebook validation outcomes are recorded in [VALIDATION.md](../VALIDATION.md).
