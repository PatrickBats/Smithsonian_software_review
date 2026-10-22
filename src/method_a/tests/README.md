# Method A code checks

These are automated software checks, unrelated to the reserved scientific test
split. They load no test-set images and do not train or evaluate a model.

Run from the repository root:

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s src/method_a/tests -v
```

Expected output ends with `Ran 27 tests` and `OK`.

| File | Inputs and behavior checked |
| --- | --- |
| [test_method_a_variants.py](test_method_a_variants.py) | 15 original checks of numerical fingerprints, recorded counts, configuration identity and output boundaries |
| [test_package.py](test_package.py) | 12 checks of four-configuration dispatch, input binding, invalid argument rejection, isolated imports, scratch directories, category metadata and CSV export |

Inputs are package code, [saved results](../../../results/method_a/README.md),
[input metadata](../../../data/method_a/README.md), and temporary fixtures created
by the tests. No external images, weights or GPU are needed. The command-help
and result-export checks execute from a different working directory.

The repository also retains three shared checks in its top-level tests folder;
CI explicitly runs those and this suite (30 total). These local checks ran on
macOS/Python 3.13. They do not establish GPU or native Windows support; some
historical filesystem-boundary checks exercise POSIX symlink behavior.
