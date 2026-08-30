# Vision Transformer on MNIST, from Scratch

This repository is an educational project for building a small Vision
Transformer (ViT) that classifies MNIST digits. The model will be implemented
from basic PyTorch tensor operations so that the complete forward pass can be
understood and inspected one step at a time.

The repository now contains the explicit ViT, MNIST data loaders, a
trained single-image debugger, and a straightforward CPU training pipeline.

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

## Baseline architecture

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

## Project structure

```text
vit-mnist-from-scratch/
|-- AGENTS.md
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- train.py
|-- src/
|   `-- vit_mnist/
|       |-- model.py
|       `-- data.py
|-- scripts/
|   `-- debug_single_image.py
|-- tests/
|   `-- test_training.py
`-- data/                       # downloaded data; ignored by Git
```

Keeping the model components together in one `model.py` makes it easy to step
from patch extraction through attention and classification without jumping
between many files. Data loading and training remain separate so the model has
no import-time side effects.

## Forward pass

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

PowerShell activation command:

```powershell
.\.venv\Scripts\Activate.ps1
```

The pinned CPU environment provides PyTorch and torchvision; torchvision is
used only for MNIST loading and basic tensor normalization.

## Results

The normal five-epoch MNIST experiment produced:

| Epoch | Train loss | Train accuracy | Test accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 0.8896 | 71.64% | 85.17% |
| 2 | 0.3598 | 88.92% | 91.83% |
| 3 | 0.2358 | 92.76% | 93.83% |
| 4 | 0.1839 | 94.36% | 94.52% |
| 5 | 0.1501 | 95.41% | 95.23% |

The tiny-subset validation used 128 fixed training examples and reached 100%
training accuracy, with a final loss of 0.0632 after 100 epochs. Deliberately
overfitting this small subset verifies that the model, loss, optimizer, and
gradient-update pipeline can learn. The final 95.23% test accuracy from normal
training demonstrates generalization to unseen MNIST test images.

## Running the project

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the normal five-epoch MNIST training experiment with:

```powershell
python .\train.py
```

Step the fixed MNIST test image at dataset index 0 through the trained model on
CPU with:

```powershell
python .\scripts\debug_single_image.py
```

The debugger loads `checkpoints/vit_mnist.pt`, keeps the model output as raw
logits, and uses batch size one with gradient recording disabled. Checkpoint
files ending in `.pt` are intentionally ignored by Git and must not be
committed.
