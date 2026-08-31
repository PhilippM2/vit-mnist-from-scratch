# Architecture: Small Vision Transformer for MNIST

## 1. Purpose

This repository implements a compact Vision Transformer (ViT) for MNIST with
PyTorch. It is designed for learning and debugging: important tensor
transformations are written as separate, named operations so they can be
inspected during a forward pass.

The implementation accepts image batches shaped `[B, 1, 28, 28]` and returns
raw classification logits shaped `[B, 10]`. Patch extraction, the query, key,
and value projections, attention scaling, softmax, and value aggregation are
all explicit.

The model intentionally does not use:

- `timm`;
- torchvision's `VisionTransformer`;
- `torch.nn.MultiheadAttention`;
- fused scaled-dot-product attention;
- a combined QKV projection;
- `Conv2d` for patch embedding; or
- code copied or adapted from an external ViT implementation.

Torchvision is used only to load MNIST and apply basic image transforms.

## 2. Shape notation

| Symbol | Meaning | Baseline value |
| --- | --- | ---: |
| `B` | Batch size | Variable |
| `C` | Image channels | `1` |
| `H`, `W` | Image height and width | `28`, `28` |
| `P` | Square patch size | `7` |
| `N` | Number of image patches, `(H/P) * (W/P)` | `16` |
| `T` | Tokens after adding the class token, `N + 1` | `17` |
| `D` | Token embedding dimension | `64` |
| `A` | Attention heads | `4` |
| `Dh` | Dimension per head, `D/A` | `16` |
| `M` | Transformer MLP hidden dimension | `128` |
| `K` | Digit classes | `10` |

## 3. Baseline configuration

The constructor defaults implement this deliberately small architecture:

- non-overlapping `7 x 7` patches on native `28 x 28` MNIST images;
- a `4 x 4` patch grid and therefore 16 image tokens;
- one learned class token and learned absolute positional embeddings;
- embedding dimension 64;
- four attention heads of dimension 16;
- two pre-layer-normalized encoder blocks;
- a `64 -> 128 -> 64` MLP with GELU;
- MLP dropout is configurable and disabled by default; and
- a final layer normalization followed by a 10-class linear head.

These values remain simple constructor arguments rather than being hidden in a
configuration framework.

## 4. End-to-end tensor shapes

For one debugger image, `B = 1`:

| Step | General shape | Debug shape |
| --- | --- | --- |
| Input image batch | `[B, C, H, W]` | `[1, 1, 28, 28]` |
| Flattened patches | `[B, N, C*P*P]` | `[1, 16, 49]` |
| Patch embeddings | `[B, N, D]` | `[1, 16, 64]` |
| Add class token | `[B, T, D]` | `[1, 17, 64]` |
| Add positional embedding | `[B, T, D]` | `[1, 17, 64]` |
| Normalized tokens | `[B, T, D]` | `[1, 17, 64]` |
| Q, K, V before head split | `[B, T, D]` each | `[1, 17, 64]` each |
| Q, K, V after head split | `[B, A, T, Dh]` each | `[1, 4, 17, 16]` each |
| Attention scores | `[B, A, T, T]` | `[1, 4, 17, 17]` |
| Attention probabilities | `[B, A, T, T]` | `[1, 4, 17, 17]` |
| Per-head context | `[B, A, T, Dh]` | `[1, 4, 17, 16]` |
| Merged attention output | `[B, T, D]` | `[1, 17, 64]` |
| Encoder output | `[B, T, D]` | `[1, 17, 64]` |
| Selected class token | `[B, D]` | `[1, 64]` |
| Class logits | `[B, K]` | `[1, 10]` |

## 5. Model components

All model components are ordered from low-level tensor operations to the
complete model in `src/vit_mnist/model.py`. Keeping them together makes a
debugger step through the forward pass without jumping between many files.

### 5.1 Patch extraction and embedding

`PatchEmbedding` validates `[B, C, H, W]` input and uses two visible
`Tensor.unfold` calls to split each image into non-overlapping spatial regions:

```text
[B, C, H, W]
    -> [B, C, H/P, W/P, P, P]
    -> [B, H/P, W/P, C, P, P]
    -> [B, N, C*P*P]
```

For the baseline this produces `[B, 16, 49]`. The grid is permuted before it is
flattened so patch tokens appear in row-major image order.

Every flattened patch is then projected through the same learned linear layer:

```text
e_n = p_n W_patch + b_patch
```

This changes `[B, 16, 49]` into `[B, 16, 64]`.

A `Conv2d` with kernel size and stride equal to the patch size is a common,
compact alternative. It is not used here because it combines extraction and
projection and hides the flattened patch vectors. `torch.nn.functional.unfold`
would also work, but the two tensor `unfold` calls expose both spatial axes.

### 5.2 Class token

The learned `class_token` parameter has shape `[1, 1, D]`. It is expanded over
the batch and prepended to the patch sequence:

```text
z_0 = concat(expand(class_token, B), patch_embeddings, token_axis)
```

The operation changes `[B, N, D]` into `[B, N+1, D]`. Expansion creates a batch
view of one shared learned parameter; it does not create independently learned
tokens for individual images.

Mean-pooling the final patch tokens is a valid alternative. The class token is
used because it makes the information-aggregation path especially visible.

### 5.3 Positional embeddings

Self-attention alone is permutation-equivariant, so it does not know where a
patch originated. A learned positional table shaped `[1, T, D]` is added to the
token sequence:

```text
z = z_0 + positional_embedding
```

Broadcasting shares the positional table across the batch. Fixed sine/cosine,
two-dimensional row/column, or relative-position encodings are reasonable
alternatives, but they add concepts that are unnecessary for this fixed
17-token example.

### 5.4 Layer normalization

Each encoder block uses pre-layer normalization. `LayerNorm` normalizes every
token independently over its final `D` features:

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

Using PyTorch's `nn.LayerNorm` keeps the implementation focused on the attention
calculation. Batch normalization is a poor fit because it mixes batch
statistics and behaves awkwardly for single-image debugging. Post-normalization
is valid, but pre-normalization gives the clear residual form used below.

### 5.5 Explicit multi-head self-attention

`MultiHeadSelfAttention` accepts and returns `[B, T, D]`. It implements the
complete attention calculation as named operations.

First, three separate learned layers create query, key, and value tensors:

```text
Q = x W_Q + b_Q
K = x W_K + b_K
V = x W_V + b_V
```

Each result has shape `[B, T, D]`. Each tensor is reshaped and transposed into
attention heads:

```text
[B, T, D] -> [B, T, A, Dh] -> [B, A, T, Dh]
```

Compatibility scores are calculated explicitly:

```text
scores = Q @ transpose(K, -2, -1)
scaled_scores = scores / sqrt(Dh)
```

The score tensor has shape `[B, A, T, T]`. Dividing by `sqrt(Dh)` prevents dot
products from growing too large as the head dimension increases.

Softmax normalizes every query row across key positions:

```text
probabilities = softmax(scaled_scores, dim=-1)
```

The implementation checks that these rows sum to approximately one. Finally,
the probabilities aggregate the values:

```text
context = probabilities @ V
```

Context has shape `[B, A, T, Dh]`. The heads are transposed, reshaped back to
`[B, T, D]`, and passed through a final output projection.

`nn.MultiheadAttention`, fused scaled-dot-product attention, and a combined
`Linear(D, 3D)` projection would be more compact. They are deliberately avoided
because they hide the core learning target: separate Q/K/V projections, head
splitting, score scaling, softmax, aggregation, and head merging.

### 5.6 Transformer MLP

`FeedForwardMLP` applies the same nonlinear transformation independently to
every token:

```text
MLP(x) = dropout(linear_2(dropout(GELU(linear_1(x)))))
```

The baseline dimensions are `64 -> 128 -> 64`, so both input and output have
shape `[B, T, D]`. A four-times expansion is common in larger ViTs; the smaller
two-times expansion keeps this model easy to inspect and train on CPU.

### 5.7 Pre-normalized encoder block

`TransformerBlock` combines attention and the MLP with residual connections:

```text
x_attention = x + Attention(LayerNorm_1(x))
x_output = x_attention + MLP(LayerNorm_2(x_attention))
```

Both sublayers preserve `[B, T, D]`, making each residual addition valid. The
baseline has two independent blocks. Stochastic depth, parameter sharing, and
deeper stacks are omitted because they distract from the small educational
baseline.

### 5.8 Final normalization and classification

After the block stack, a final layer normalization preserves `[B, T, D]`. Token
position zero is selected as the image representation:

```text
class_representation = normalized_tokens[:, 0, :]
logits = classifier(class_representation)
```

This changes `[B, T, D]` to `[B, D]` and then `[B, K]`. The model returns raw
logits. It does not apply softmax because `CrossEntropyLoss` expects logits.

## 6. Data preprocessing

`src/vit_mnist/data.py` uses torchvision to download and load separate MNIST
training and test datasets. `ToTensor` produces `[1, 28, 28]` floating-point
images, and `Normalize` applies the documented MNIST mean `0.1307` and standard
deviation `0.3081`.

Dataset construction and downloads happen only when `create_mnist_loaders` is
called, never at import time. Data is stored below the ignored `data/`
directory. Data loaders default to `num_workers=0`, which is predictable on
Windows and convenient for debugging.

Padding to `32 x 32` and using `4 x 4` patches is a common alternative. The
baseline keeps native MNIST dimensions and uses `7 x 7` patches so the
attention matrix remains only `17 x 17` per head.

## 7. Training and checkpoints

`train.py` provides a deliberately direct CPU training loop:

- `CrossEntropyLoss` consumes raw `[B, 10]` logits;
- AdamW uses learning rate `3e-4` and weight decay `1e-2`;
- batch size is 128;
- normal training defaults to five epochs;
- Python and PyTorch use deterministic seed 0;
- training loss and accuracy are printed every epoch;
- test accuracy is printed every normal-training epoch; and
- the final state dictionary is saved by default to
  `checkpoints/vit_mnist.pt`.

The `--tiny-subset` mode selects the first 128 training examples and makes it
possible to deliberately overfit them as an integration check. Command-line
options can override the epoch count, data root, checkpoint path, initial
checkpoint, and whether downloading is allowed.

The training code selects CPU explicitly. It does not include mixed precision,
distributed training, compilation, learning-rate scheduling, data augmentation,
or experiment tracking.

## 8. Debugging one image

`scripts/debug_single_image.py` performs a deterministic inference-only pass on
MNIST test image index 0. It requires the default trained checkpoint at
`checkpoints/vit_mnist.pt`, loads it on CPU, constructs a batch shaped
`[1, 1, 28, 28]`, and prints the true label, raw logits, predicted label, and
whether the prediction is correct.

Useful breakpoint locations in `src/vit_mnist/model.py` are:

1. after `flattened_patches` is created;
2. after `patch_embeddings` is projected;
3. after the class token and positional embedding are added;
4. after the separate `queries`, `keys`, and `values` projections;
5. after the tensors are split into heads;
6. after `attention_scores` and `scaled_attention_scores` are calculated;
7. after `attention_probabilities` is calculated;
8. after `context` aggregation and the head merge;
9. after each residual addition; and
10. before and after the classification head.

At these points, local variables follow the shapes in Section 4. The script
runs directly from PowerShell or through the VS Code Python debugger. No
repository-specific VS Code configuration is required.

## 9. Validation that exists in the repository

The implementation uses clear constructor errors and forward-pass assertions
for important invariants, including image dimensions, patch divisibility,
embedding/head divisibility, component tensor shapes, attention probability row
sums, and output shapes.

`tests/test_model.py` checks patch order with a hand-verifiable image, component
and end-to-end shapes, finite forward values and gradients, and supported
invalid configurations. Attention probabilities are local debugger variables
rather than part of the component's return value, so the test suite exercises
the existing in-forward probability-row assertion instead of adding a new
production API.

`tests/test_training.py` checks that one training batch updates parameters,
evaluation returns accuracy without creating gradients, and a saved state
dictionary round trip preserves logits.

Running `src/vit_mnist/model.py` directly executes an additional synthetic
component smoke test without downloading MNIST. The tiny-subset training mode
provides the data-pipeline integration check; it is intentionally not part of
the fast unit-test suite.

## 10. Deliberate scope

The repository remains focused on an understandable baseline. Performance
optimization, compilation, mixed precision, distributed training, attention
visualization UI, architecture search, larger datasets, and external experiment
tracking are outside its current scope.
