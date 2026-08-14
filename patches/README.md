# llama.cpp patches

The three commits that make MoE expert-offload fast on this rig, exported from
`github.com/thecodacus/llama.cpp` branch `fable5/prefetch-experts`.

They are vendored here so the build is reproducible from this repo alone, without
depending on that fork staying available. See the main [README](../README.md) for what
each patch does.

- `0001` — pin mmap-backed CPU weights for faster H2D uploads (`GGML_CUDA_REGISTER_HOST=1`)
- `0002` — overlap offloaded expert weight uploads with compute (`GGML_SCHED_PREFETCH_EXPERTS`)
- `0003` — size prefetch slots per layer, fix a fallback use-after-free

Authored by `thecodacus <thecodacus@gmail.com>`. `git am` preserves that authorship.

## Base commit

These apply cleanly to upstream llama.cpp at:

```
4fc4ec5541b243957ae5099edb67372f8f3b550e
```

## Applying to a fresh clone

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout -b prefetch-experts 4fc4ec5541b243957ae5099edb67372f8f3b550e
git am ..\patches\*.patch
```

## Re-exporting after new work

From the fork, with the three commits on top of an upstream base:

```powershell
git -C llama.cpp format-patch -3 --no-numbered --zero-commit --output-directory ..\patches
```

Then update the base commit hash above to the new `git merge-base HEAD origin/master`.

## Rebasing onto newer upstream

Conflicts will land in `ggml/src/ggml-backend.cpp` around
`ggml_backend_sched_compute_splits()`, which is a hot file upstream.
