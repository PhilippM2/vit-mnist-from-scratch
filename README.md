# Vision Transformer on MNIST, from Scratch

This repository contains an educational Vision Transformer (ViT) that
classifies MNIST digits. The model is written from basic PyTorch tensor
operations so the complete forward pass can be understood and inspected one
step at a time.

Readability and debuggability take priority over training speed and benchmark
performance. In particular, the implementation keeps patch extraction and the
query, key, value, attention-score, softmax, and context calculations explicit.

## Learning goals and constraints

- Follow how an image becomes a sequence of patch tokens.
- Inspect separate query, key, and value projections.
- Inspect multi-head scaled dot-product self-attention.
- Understand residual connections, layer normalization, and the transformer
  MLP.
- Train a deliberately small ViT on MNIST.
- Step through one image in the VS Code debugger.

The implementation does not use:

- `timm`
- torchvision's `VisionTransformer`
- `torch.nn.MultiheadAttention`
- fused or external ViT implementations

Torchvision is used only for MNIST loading and basic image transforms.

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

Patch embedding is deliberately explicit: the code splits each image into
non-overlapping `7 x 7` patches, flattens every grayscale patch from
`1 x 7 x 7` to 49 values, and applies `nn.Linear(49, 64)`. A `Conv2d` patch
projection is intentionally excluded so every rearrangement remains visible in
the debugger.

## Forward-pass shape trace

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
[B, 10] raw digit logits
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component contracts, equations,
design alternatives, test coverage, and detailed debugger guidance.

## Project structure

```text
vit-mnist-from-scratch/
|-- ARCHITECTURE.md
|-- README.md
|-- requirements.txt
|-- train.py
|-- src/
|   `-- vit_mnist/
|       |-- model.py
|       `-- data.py
|-- scripts/
|   `-- debug_single_image.py
|-- tests/
|   |-- test_model.py
|   `-- test_training.py
|-- data/                       # generated/downloaded; ignored by Git
`-- checkpoints/                # generated; ignored by Git
```

Keeping the model components together in `model.py` makes it possible to step
from patch extraction through attention and classification without jumping
between many files. Data loading and training remain separate, so importing the
model does not download data, start training, or select a device.

## Setup

Python 3.12 is the environment used for the reported runs. From a fresh clone,
create and activate a virtual environment and install the pinned dependencies.
On Windows PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train, then debug

### 1. Train the model

```powershell
python train.py
```

Normal training runs on CPU for five epochs. It downloads MNIST into `data/` if
necessary, prints training loss and accuracy plus test accuracy for each epoch,
and writes the final state dictionary to `checkpoints/vit_mnist.pt`. Both the
dataset and checkpoint are ignored by Git.

Use `--no-download` to require an existing local MNIST dataset. Run
`python train.py --help` for the small set of checkpoint, data-root, epoch,
and tiny-subset options.

### 2. Debug one trained prediction

After the default checkpoint exists, run:

```powershell
python scripts/debug_single_image.py
```

The script loads `checkpoints/vit_mnist.pt` on CPU and passes MNIST test image
index 0 through the model with batch shape `[1, 1, 28, 28]`. It prints the true
label, raw logits, prediction, and correctness.

Useful breakpoint locations in `src/vit_mnist/model.py` include patch
extraction, patch projection, separate Q/K/V projection, head splitting,
attention scores, attention probabilities, context aggregation, residual
additions, and final logits. The precise breakpoint sequence and expected local
tensor shapes are in [ARCHITECTURE.md](ARCHITECTURE.md#8-debugging-one-image).

## Tests and synthetic smoke test

Run the unit tests without downloading MNIST:

```powershell
python -m unittest discover -s tests
```

Run the model's additional synthetic component smoke test with:

```powershell
python src/vit_mnist/model.py
```

The unit tests cover patch order and shapes, model component contracts, finite
forward and backward values, supported invalid configurations, one training
update, evaluation behavior, and checkpoint round trips.

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
training accuracy, with a final loss of 0.0632 after 100 epochs. That historical
run used:

```powershell
python train.py --tiny-subset --epochs 100 --checkpoint-path checkpoints/tiny-validation.pt
```

Deliberately overfitting this subset checks that the model, loss, optimizer,
and gradient-update pipeline can learn. The final 95.23% test accuracy from the
normal run demonstrates generalization to unseen MNIST test images.
