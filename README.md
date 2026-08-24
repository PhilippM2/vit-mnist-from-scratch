# Vision Transformer on MNIST, from Scratch

This repository is an educational project for building a small Vision
Transformer (ViT) that classifies MNIST digits. The model will be implemented
from basic PyTorch tensor operations so that the complete forward pass can be
understood and inspected one step at a time.

The project currently contains planning documentation only. No model code or
third-party dependencies have been added yet.

## Goals

- Understand how an image becomes a sequence of patch tokens.
- Implement query, key, and value projections explicitly.
- Implement multi-head scaled dot-product self-attention explicitly.
- Understand residual connections, layer normalization, and the transformer
  MLP.
- Train a deliberately small ViT on MNIST.
- Make a single-image forward pass easy to follow in the VS Code debugger.

Readability and debuggability take priority over training speed and benchmark
performance.

## Constraints

The implementation will not use:

- `timm`
- torchvision's `VisionTransformer`
- `torch.nn.MultiheadAttention`
- external ViT implementations

Torchvision may be used later for loading MNIST and applying basic image
transforms. The model itself will be written locally.

## Proposed baseline

| Setting | Value |
| --- | ---: |
| Input | `1 x 28 x 28` |
| Patch size | `7 x 7` |
| Image patches | `16` |
| Tokens with class token | `17` |
| Embedding dimension | `64` |
| Attention heads | `4` |
| Dimension per head | `16` |
| Encoder blocks | `2` |
| MLP hidden dimension | `128` |
| Output classes | `10` |

This baseline is small enough for shape inspection and CPU experiments while
still containing the important ViT mechanisms.

Patch embedding will be deliberately explicit: split the image into
non-overlapping `7 x 7` patches, flatten each grayscale patch from `1 x 7 x 7`
to `49` values, and apply `nn.Linear(49, 64)` to each patch. A `Conv2d` patch
projection is intentionally excluded from the baseline so every rearrangement
remains visible in the debugger.

## Proposed minimal structure

Only the three documentation files at the top are created during the planning
phase. The remaining files are planned and will be added incrementally after
the plan is approved.

```text
vit-mnist-from-scratch/
|-- AGENTS.md
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- src/
|   `-- vit_mnist/
|       |-- __init__.py
|       |-- model.py
|       |-- data.py
|       `-- train.py
|-- scripts/
|   `-- debug_single_image.py
|-- tests/
|   |-- test_attention.py
|   `-- test_model.py
`-- data/                       # downloaded data; ignored by Git
```

Keeping the model components together in one `model.py` makes it easy to step
from patch extraction through attention and classification without jumping
between many files. Data loading and training remain separate so the model has
no import-time side effects.

## Planned forward pass

```text
[B, 1, 28, 28] image batch
        |
        v
[B, 16, 49] flattened patches
        |
        v
[B, 16, 64] patch embeddings
        |
        v
[B, 17, 64] class token + positional embeddings
        |
        v
[B, 17, 64] two transformer encoder blocks
        |
        v
[B, 64] class-token representation
        |
        v
[B, 10] digit logits
```

The complete component contracts, equations, alternatives, tests, and delivery
phases are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Environment

The repository currently uses a Python 3.12 virtual environment at `.venv`.
No packages are installed as part of the planning phase.

PowerShell activation command:

```powershell
.\.venv\Scripts\Activate.ps1
```

Dependency selection and installation will happen only after the architecture
plan is reviewed.
