# Implementation Plan: Small Vision Transformer for MNIST

## 1. Objective and acceptance criteria

Build a compact Vision Transformer in PyTorch for MNIST while keeping every
important operation visible. The first complete version is successful when:

- it accepts a batch shaped `[B, 1, 28, 28]` and returns logits shaped `[B, 10]`;
- patchification, Q/K/V projections, attention scaling, softmax, and value
  aggregation are implemented explicitly;
- all component shape contracts are covered by tests;
- one image can be stepped through using `scripts/debug_single_image.py` and
  `num_workers=0` on Windows;
- a tiny MNIST subset can be deliberately overfit as an integration check;
- the normal training path can train and evaluate the model on MNIST;
- no prohibited model implementation or attention abstraction is used.

This architecture is the approved planning baseline. Implementation proceeds
one reviewed component at a time; each component is tested and explained before
work starts on the next one.

## 2. Shape notation

| Symbol | Meaning | Baseline value |
| --- | --- | ---: |
| `B` | batch size | variable |
| `C` | image channels | `1` |
| `H`, `W` | image height and width | `28`, `28` |
| `P` | square patch size | `7` |
| `N` | number of image patches, `(H/P) * (W/P)` | `16` |
| `T` | sequence length after adding a class token, `N + 1` | `17` |
| `D` | token embedding dimension | `64` |
| `A` | number of attention heads | `4` |
| `Dh` | dimension per head, `D/A` | `16` |
| `M` | hidden dimension of the transformer MLP | `128` |
| `K` | number of digit classes | `10` |

## 3. Baseline architecture decisions

- Use non-overlapping `7 x 7` patches. MNIST becomes a `4 x 4` patch grid,
  producing only 16 image tokens. This makes a `17 x 17` attention matrix easy
  to inspect.
- Use a learned class token and learned absolute positional embeddings.
- Use two pre-layer-normalized encoder blocks.
- Use four attention heads with separate Q, K, and V `Linear(D, D)` layers.
- Use an MLP expansion ratio of two: `64 -> 128 -> 64`.
- Use GELU in the MLP.
- Start with dropout configurable and set to `0.0` for debugger/tests. A small
  value such as `0.1` can be enabled for normal training if needed.
- Apply a final layer normalization before the classification head.

These values are constructor defaults, not global constants.

## 4. End-to-end shape trace

For one debugger image, `B = 1`:

| Step | General shape | Debug shape |
| --- | --- | --- |
| Input image batch | `[B, C, H, W]` | `[1, 1, 28, 28]` |
| Flattened patches | `[B, N, C*P*P]` | `[1, 16, 49]` |
| Patch embeddings | `[B, N, D]` | `[1, 16, 64]` |
| Add class token | `[B, T, D]` | `[1, 17, 64]` |
| Add position embedding | `[B, T, D]` | `[1, 17, 64]` |
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

## 5. Planned model components

### 5.1 Input validation and patch extraction

**Purpose**

Validate the expected image layout and convert each image into a sequence of
non-overlapping flattened patches.

**Input shape**

`[B, C, H, W]`, baseline `[B, 1, 28, 28]`.

**Output shape**

`[B, N, C*P*P]`, baseline `[B, 16, 49]`.

**Mathematical operation**

Split the spatial axes into a grid of `P x P` regions. If patch index `n`
corresponds to grid location `(r, c)`, its flattened values are:

```text
patch[n] = flatten(x[:, :, r*P:(r+1)*P, c*P:(c+1)*P])
```

The implementation uses `torch.Tensor.unfold` twice, followed by an explicit
dimension permutation and reshape. This must visibly produce
`[B, 16, 1, 7, 7]` patches and then `[B, 16, 49]` flattened patches before the
linear projection. Assertions verify `H % P == 0` and `W % P == 0`.

**Relevant alternatives**

- A `Conv2d` with kernel size and stride equal to the patch size can combine
  patch extraction and projection. It is compact and fast, but hides the patch
  vectors, so it is explicitly excluded from the baseline implementation.
- `torch.nn.functional.unfold` is also valid, but two visible tensor `unfold`
  calls make the spatial transformation easier to inspect.
- Pixel-level tokens (`P=1`) preserve maximum detail but produce 785 tokens
  including the class token, making attention much harder to debug.

### 5.2 Linear patch embedding

**Purpose**

Map every flattened patch into the shared transformer embedding space.

**Input shape**

`[B, N, C*P*P]`, baseline `[B, 16, 49]`.

**Output shape**

`[B, N, D]`, baseline `[B, 16, 64]`.

**Mathematical operation**

For each patch vector `p_n`:

```text
e_n = p_n W_patch + b_patch
```

where `W_patch` has shape `[C*P*P, D]` and the same learned projection is used
for every patch.

**Relevant alternatives**

- A strided convolution is the common compact formulation but is less explicit.
- Leaving patches in their raw 49-dimensional space would remove a useful
  configurable embedding space and make head divisibility awkward.
- Normalizing individual patch vectors is possible but not necessary for the
  first version because image normalization and transformer layer normalization
  already provide stable inputs.

### 5.3 Learnable class token

**Purpose**

Provide one sequence position that can aggregate information from every image
patch and later serve as the image representation.

**Input shape**

Patch embeddings `[B, N, D]`; parameter `class_token` shaped `[1, 1, D]`.

**Output shape**

`[B, N+1, D]`, baseline `[B, 17, 64]`.

**Mathematical operation**

Expand the same learned token across the batch and concatenate it before the
patch tokens:

```text
z_0 = concat(expand(class_token, B), patch_embeddings, token_axis)
```

Expansion must not create independently learned class tokens per batch item.

**Relevant alternatives**

- Mean-pool all final patch tokens. This removes a parameter and often works,
  but the class token makes the ViT information flow more explicit.
- Max pooling is simple but discards more distributed evidence.
- Appending rather than prepending the class token is mathematically valid if
  position handling and token selection stay consistent; prepending follows the
  conventional layout.

### 5.4 Learnable positional embedding

**Purpose**

Encode token order and patch location because self-attention alone is
permutation-equivariant.

**Input shape**

Token sequence `[B, T, D]`; parameter `position_embedding` shaped `[1, T, D]`.

**Output shape**

`[B, T, D]`.

**Mathematical operation**

Add a learned vector to each sequence position:

```text
z = z_0 + position_embedding
```

Broadcasting repeats the positional table across the batch only.

**Relevant alternatives**

- Fixed sine/cosine embeddings avoid learned position parameters and can make
  extrapolation easier, but are less direct for a fixed 17-token exercise.
- Two-dimensional row and column embeddings expose the patch grid structure,
  but add concepts that are not necessary for the baseline.
- Relative position bias is useful in larger vision models but complicates the
  attention score calculation.

### 5.5 Layer normalization

**Purpose**

Normalize each token's feature vector before attention and before the MLP,
improving optimization while preserving batch independence.

**Input shape**

`[B, T, D]`.

**Output shape**

`[B, T, D]`.

**Mathematical operation**

For each token vector `x`:

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

The mean and variance are computed over the final feature dimension `D`.
`torch.nn.LayerNorm` is allowed because the learning target is attention rather
than reimplementing numerical normalization.

**Relevant alternatives**

- Post-normalization applies layer normalization after each residual addition.
  The baseline uses pre-normalization because it is typically easier to train
  and gives a clear `residual + sublayer(norm(x))` pattern.
- Batch normalization mixes batch statistics and is a poor fit for token
  sequences and single-image debugging.
- RMS normalization is simpler but is not needed for the initial ViT lesson.

### 5.6 Explicit multi-head self-attention

**Purpose**

Allow every token to gather a content-dependent weighted combination of all
tokens, using several attention heads to represent different relationships.

**Input shape**

`x` shaped `[B, T, D]`, baseline `[B, 17, 64]`.

**Output shape**

`[B, T, D]`, baseline `[B, 17, 64]`.

**Mathematical operation**

1. Apply three separate learned projections:

   ```text
   Q = x W_Q + b_Q
   K = x W_K + b_K
   V = x W_V + b_V
   ```

   Each result has shape `[B, T, D]`.

2. Reshape and transpose each tensor into heads:

   ```text
   [B, T, D] -> [B, T, A, Dh] -> [B, A, T, Dh]
   ```

3. Compute scaled pairwise compatibility scores:

   ```text
   scores = Q @ transpose(K, -2, -1) / sqrt(Dh)
   ```

   The result has shape `[B, A, T, T]`.

4. Normalize each query row across key positions:

   ```text
   probabilities = softmax(scores, dim=-1)
   ```

5. Aggregate the values:

   ```text
   context = probabilities @ V
   ```

   The result has shape `[B, A, T, Dh]`.

6. Transpose and reshape the heads back to `[B, T, D]`, then apply an output
   projection `W_O`.

The code should retain names such as `queries`, `keys`, `values`,
`attention_scores`, `attention_probabilities`, and `context` so debugger
breakpoints expose the complete calculation.

**Relevant alternatives**

- `torch.nn.MultiheadAttention` and fused scaled-dot-product attention are
  prohibited because they hide the learning target.
- A combined `Linear(D, 3D)` QKV projection is faster but makes Q, K, and V less
  explicit. Three separate layers are the baseline.
- Single-head attention is simpler but does not demonstrate the head split and
  merge operations.
- Attention masks are unnecessary because all image tokens may attend to one
  another. Mask support should not be added until a use case requires it.
- Attention dropout may be applied after softmax for training, but should be
  disabled in the debugger.

### 5.7 Transformer MLP

**Purpose**

Apply a learned nonlinear transformation independently to every token after the
attention sublayer.

**Input shape**

`[B, T, D]`, baseline `[B, 17, 64]`.

**Output shape**

`[B, T, D]`, baseline `[B, 17, 64]`.

**Mathematical operation**

```text
MLP(x) = dropout(linear_2(dropout(GELU(linear_1(x)))))
```

with `linear_1: D -> M` and `linear_2: M -> D`, baseline
`64 -> 128 -> 64`. Linear layers operate independently on the final dimension
at every token position.

**Relevant alternatives**

- ReLU is easier to visualize but GELU is conventional in transformers and
  remains readable.
- A four-times expansion (`M=4D`) is common in larger ViTs; `M=2D` keeps the
  educational model small.
- Gated MLP variants can improve performance but add parameters and operations
  unrelated to the core lesson.

### 5.8 Transformer encoder block

**Purpose**

Combine self-attention and the token-wise MLP with residual connections.

**Input shape**

`[B, T, D]`.

**Output shape**

`[B, T, D]`.

**Mathematical operation**

Use pre-layer normalization:

```text
x_attention = x + Attention(LayerNorm_1(x))
x_output    = x_attention + MLP(LayerNorm_2(x_attention))
```

Both sublayers preserve `[B, T, D]`, which makes the residual additions valid.

**Relevant alternatives**

- Post-normalization uses `LayerNorm(x + sublayer(x))`; it follows the original
  Transformer presentation but can be harder to optimize.
- Stochastic depth is useful for deep networks but unnecessary for two blocks.
- Sharing one block's parameters across depth would reduce parameters but is not
  the standard ViT structure and makes the conceptual stack less direct.

### 5.9 Encoder stack and final normalization

**Purpose**

Apply multiple encoder blocks sequentially and normalize the final token
representations.

**Input shape**

`[B, T, D]`.

**Output shape**

`[B, T, D]`.

**Mathematical operation**

For blocks `l = 1 ... L`:

```text
x_l = EncoderBlock_l(x_(l-1))
encoded = LayerNorm_final(x_L)
```

The baseline uses `L=2`, with independent parameters in each block.

**Relevant alternatives**

- One block is even easier to inspect but provides little sense of hierarchical
  refinement.
- More blocks may improve capacity but slow CPU debugging and obscure the small
  model's behavior.
- Omitting final normalization is possible, but final normalization is a clear
  and common companion to pre-normalized blocks.

### 5.10 Classification head

**Purpose**

Convert the final class-token representation into one logit per MNIST digit.

**Input shape**

Encoded tokens `[B, T, D]`; selected class token `[B, D]`.

**Output shape**

`[B, K]`, baseline `[B, 10]`.

**Mathematical operation**

Select sequence position zero and apply a linear map:

```text
class_representation = encoded[:, 0, :]
logits = class_representation W_head + b_head
```

The model returns raw logits. Softmax is not applied inside the model because
cross-entropy loss expects logits.

**Relevant alternatives**

- Mean pooling all patch tokens avoids a class token but changes the planned
  information-aggregation mechanism.
- A two-layer classification MLP adds capacity but distracts from the encoder.
- Applying softmax in the model is convenient for display but is numerically and
  conceptually wrong for the planned training loss.

### 5.11 Complete Vision Transformer module

**Purpose**

Compose all components into one transparent forward path and own the learned
class and positional parameters.

**Input shape**

`[B, 1, 28, 28]`.

**Output shape**

`[B, 10]` raw logits.

**Mathematical operation**

```text
patches  = patchify(images)
tokens   = patch_projection(patches)
tokens   = prepend_class_token(tokens)
tokens   = tokens + position_embedding
encoded  = final_norm(encoder_blocks(tokens))
logits   = classifier(encoded[:, 0, :])
```

The constructor checks the architectural invariants once. The forward method
checks the runtime image shape and then names each intermediate value.

**Relevant alternatives**

- Splitting each small component into its own file improves reuse but makes the
  debugger jump between files. One ordered model file is preferred here.
- Returning intermediate tensors from every normal forward pass complicates the
  training API. The debugger can inspect locals; an optional, explicit debug
  method may be considered only if inspection proves inconvenient.
- A configuration library is unnecessary. Plain constructor arguments are
  easier to understand.

## 6. Data and preprocessing plan

After approval and dependency installation:

- Use torchvision only for `datasets.MNIST` and basic transforms.
- Convert images to tensors shaped `[1, 28, 28]`.
- Normalize using a clearly documented MNIST mean and standard deviation, or
  begin with simple `[0, 1]` tensors and add normalization after a baseline run.
- Keep downloads under ignored `data/`.
- Use `num_workers=0` by default on Windows for predictable debugging.
- Separate train and test loaders; do not download data during module import.

Design alternatives include padding MNIST to `32 x 32` and using `4 x 4`
patches. The baseline retains the native `28 x 28` image and uses `7 x 7`
patches so preprocessing stays minimal and attention remains small.

## 7. Training plan

Initial, intentionally simple settings:

- objective: `CrossEntropyLoss` on raw `[B, 10]` logits;
- optimizer: AdamW;
- initial learning rate: `3e-4`;
- weight decay: `1e-2`;
- batch size: approximately `128` for normal training and `1` for debugging;
- epochs: start with `5`, then adjust based on the learning curve;
- deterministic seed for Python and PyTorch;
- device selection kept explicit (`cpu` or `cuda`) and printed once;
- report average training loss and test accuracy per epoch;
- save checkpoints only when explicitly requested; `*.pt` is already ignored.

Before a full run, deliberately overfit a tiny fixed subset. Failure to reduce
that loss is treated as an implementation bug rather than a hyperparameter
problem.

Relevant alternatives:

- SGD is easier to explain but usually needs more tuning for transformers.
- A learning-rate scheduler and warmup are common for ViTs but should be added
  only if the small baseline needs them.
- Data augmentation can improve generalization but is deferred until the plain
  pipeline is verified.

## 8. Testing plan

Use `unittest` initially to avoid another dependency.

### Component tests

1. Patch extraction returns `[B, 16, 49]` and preserves a known spatial order.
2. Patch projection returns `[B, 16, 64]`.
3. Class-token insertion and positional addition return `[B, 17, 64]`.
4. Q, K, and V split into `[B, 4, 17, 16]`.
5. Attention scores and probabilities have shape `[B, 4, 17, 17]`.
6. Every probability row sums to approximately one.
7. Attention output returns `[B, 17, 64]` and finite gradients.
8. MLP and encoder blocks preserve `[B, 17, 64]`.
9. The classifier returns `[B, 10]`.

### Failure tests

- wrong channel count;
- wrong image size;
- image size not divisible by patch size;
- embedding dimension not divisible by head count;
- invalid number of heads or encoder blocks.

### Integration tests

- fixed-seed forward pass on a synthetic batch;
- one backward pass with finite loss and gradients;
- one-image debugger entry point;
- tiny-subset overfit check;
- short CPU training smoke test.

Tests should check contracts and invariants rather than freezing arbitrary
floating-point output values.

## 9. VS Code single-image debugging plan

Create `scripts/debug_single_image.py` after the model exists. It should:

1. set deterministic seeds;
2. construct the small baseline with dropout disabled;
3. load one MNIST test image or create a synthetic fallback image;
4. add exactly one batch dimension, producing `[1, 1, 28, 28]`;
5. run the model on CPU in evaluation mode;
6. print the true label, predicted label, and logits;
7. avoid worker processes and complex command-line requirements;
8. optionally load a checkpoint if a path is supplied.

Recommended breakpoint sequence in `model.py`:

1. immediately after patch extraction;
2. after patch projection;
3. after class-token and position addition;
4. after separate Q, K, and V projections;
5. after the head split;
6. after scaled attention scores;
7. after softmax probabilities;
8. after value aggregation and head merge;
9. after each residual addition;
10. before and after the classification head.

At each stop, the local variables should match the debug shapes in Section 4.
A later `.vscode/launch.json` can point to this script if requested, but the
script must also run directly from PowerShell.

## 10. Implementation phases

### Phase 0: Review this plan

- Confirm the proposed `7 x 7` patch size and small architecture.
- Confirm the one-file model layout.
- Make any desired educational changes before dependencies or source files are
  introduced.

### Phase 1: Add the smallest required dependencies

- Add only PyTorch and torchvision after explicit approval.
- Record the environment in a minimal dependency file.
- Verify imports and device availability.

### Phase 2: Implement patch and token preparation

- Create the package skeleton and `model.py`.
- Implement patch extraction, patch projection, class token, and positional
  embedding.
- Add hand-checkable patch-order and shape tests.

### Phase 3: Implement explicit attention

- Add separate Q, K, and V projections.
- Add head split/merge steps, scaling, softmax, and value aggregation.
- Test shapes, probability row sums, and gradients.
- Inspect the attention matrix in the debugger before continuing.

### Phase 4: Implement the encoder and classifier

- Add the MLP, pre-normalized encoder block, block stack, final normalization,
  and classification head.
- Run end-to-end forward and backward tests.

### Phase 5: Add data loading and single-image debugging

- Add MNIST loaders with no import-time download.
- Add `debug_single_image.py` with batch size one and CPU defaults.
- Verify every breakpoint shape listed above.

### Phase 6: Add the training loop

- Implement a straightforward train/evaluate loop.
- First overfit a tiny fixed subset.
- Then run a short MNIST training experiment and report learning behavior.

### Phase 7: Refine documentation

- Add final run commands and observed results to the README.
- Document any approved architecture changes.
- Keep optimization or visualization work separate from the understandable
  baseline.

## 11. Explicitly deferred work

- performance optimization and compilation;
- mixed-precision or distributed training;
- attention visualization UI;
- checkpoint management beyond a simple optional file;
- architecture search;
- larger image datasets;
- external experiment tracking;
- model code of any kind before this plan is approved.
