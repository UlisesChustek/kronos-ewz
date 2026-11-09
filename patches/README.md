# Vendored Kronos patches

The repository vendors Kronos under `third_party/Kronos`. Its upstream base commit is:

```
67b630e67f6a18c9e9be918d9b4337c960db1e9a   (67b630e)
Merge pull request #243 from ElhamDevelopmentStudio/fix/batch-dimension-training
```

## Files in this directory

| File | Purpose |
|---|---|
| `vendored-kronos-from-67b630e.patch` | Cumulative `git diff` of tracked vendored files against `67b630e` |
| `patch_metadata.json` | Base commit, modified-file list, patch SHA-256, generation command |

The patch contains **all cumulative local modifications** to tracked vendored files, including
changes that predate the current research upgrade. It is not a per-session delta.

Modified tracked files:

- `finetune_csv/config_loader.py`
- `finetune_csv/finetune_base_model.py`
- `finetune_csv/train_sequential.py`
- `model/module.py`

Untracked `finetune_csv/finetune_base_model.py.orig` and `model/module.py.orig` are pre-existing
local snapshots. They are preserved in place and are **not** part of the patch.

## Regenerate

From the repository root:

```bash
cd third_party/Kronos
git diff > ../../patches/vendored-kronos-from-67b630e.patch
sha256sum ../../patches/vendored-kronos-from-67b630e.patch
```

## Reproduce from a clean checkout

```bash
git clone https://github.com/shiyu-coder/Kronos.git /tmp/Kronos-clean
cd /tmp/Kronos-clean
git checkout 67b630e67f6a18c9e9be918d9b4337c960db1e9a
git apply --check /path/to/kronos-ewz/patches/vendored-kronos-from-67b630e.patch
git apply /path/to/kronos-ewz/patches/vendored-kronos-from-67b630e.patch
```

Apply the patch only to a clean `67b630e` checkout. Applying it on top of already-modified files
will fail or duplicate changes.

## Scope note

The patch fixes data-window alignment, normalization leakage, loss-mask defaults, deterministic
teacher forcing, accumulation/early-stopping behaviour, run-path overwrite protection, and
cross-attention causality (including padding-mask handling). The accompanying tests under
`tests/` exercise these behaviours without needing a GPU or a checkpoint.
