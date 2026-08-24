# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Project intent

This is an educational implementation of a small Vision Transformer (ViT) for
MNIST. The main goal is to make every tensor transformation and mathematical
operation understandable and easy to inspect in the VS Code debugger. Prefer
clarity, explicitness, and useful assertions over speed or abstraction.

## Current phase gate

- The architecture and implementation plan have been approved.
- Implement only the component explicitly approved for the current step, test
  it, explain it, and stop before beginning the next component.
- Keep documentation synchronized with any later design changes.

## Non-negotiable implementation constraints

- Do not use `timm`.
- Do not use `torchvision.models.VisionTransformer` or another torchvision ViT
  implementation.
- Do not use `torch.nn.MultiheadAttention`.
- Do not copy or adapt an external ViT implementation.
- Implement the query, key, and value projections explicitly as separate,
  visible operations.
- Implement scaled dot-product attention explicitly with matrix
  multiplication, scaling by `sqrt(head_dim)`, softmax, and value aggregation.
- `torchvision` may later be used only for MNIST data loading and basic image
  transforms.
- Do not add performance-oriented machinery such as compilation, mixed
  precision, distributed training, or fused attention unless the educational
  baseline is already complete and the user explicitly requests it.

## Baseline architecture

Unless the reviewed plan is changed, use this intentionally small baseline:

- input: grayscale MNIST images, `28 x 28`
- patch size: `7 x 7`
- number of image patches: `16`
- patch embedding: explicit patch extraction, flatten `1 x 7 x 7` to `49`,
  then apply `Linear(49, 64)`; do not use `Conv2d` for the baseline
- embedding dimension: `64`
- attention heads: `4`
- dimension per head: `16`
- encoder blocks: `2`
- MLP hidden dimension: `128`
- class tokens: `1`
- output classes: `10`
- encoder style: pre-layer normalization

Keep these values configurable through simple constructor arguments, but do not
introduce a configuration framework.

## Code organization and style

- Keep the model components together in `src/vit_mnist/model.py`, ordered from
  low-level tensor operations to the complete model. This makes stepping through
  a forward pass straightforward.
- Use one small class per conceptual component where a class improves clarity.
- Use descriptive names such as `batch_size`, `num_tokens`, `embedding_dim`, and
  `head_dim`; avoid single-letter variable names except in displayed equations.
- Add type hints and concise docstrings that state input and output shapes.
- Put shape comments beside non-obvious `reshape`, `permute`, `transpose`, and
  matrix multiplication operations.
- Prefer `reshape`, `transpose`, and `torch.matmul` over dense one-line tensor
  expressions.
- Keep query, key, and value projections on separate lines.
- Validate important invariants with clear errors, including image size,
  divisibility by patch size, and embedding dimension divisibility by the number
  of heads.
- Avoid in-place tensor operations in the initial implementation.
- Avoid module-level side effects. Importing a module must not download data,
  start training, or select a device.
- Keep training orchestration out of the model module.

Use this shape notation consistently:

- `B`: batch size
- `C`: image channels
- `H`, `W`: image height and width
- `P`: patch size
- `N`: number of image patches
- `T`: number of tokens, including the class token
- `D`: embedding dimension
- `A`: number of attention heads
- `Dh`: dimension per attention head
- `M`: MLP hidden dimension

## Debuggability requirements

- Provide `scripts/debug_single_image.py` during implementation.
- Its model input must have shape `[1, 1, 28, 28]`.
- Use deterministic seeds and `num_workers=0` on Windows.
- Make it possible to stop at patch extraction, Q/K/V projection, attention
  scores, attention probabilities, context aggregation, and final logits.
- Keep intermediate expressions named rather than embedding them inside larger
  expressions.
- The debug script should support a forward pass without requiring a trained
  checkpoint; checkpoint loading can be optional.

## Tests required during implementation

- Test patch extraction order and shape with a hand-checkable synthetic image.
- Test every component's input/output shape contract.
- Check that attention rows sum to approximately one after softmax.
- Check that attention output and gradients are finite.
- Check that invalid shape configurations fail with useful messages.
- Run an end-to-end forward and backward pass on a tiny synthetic batch.
- Overfit a very small MNIST subset as an integration check before full training.

Use the Python standard library's `unittest` initially unless the user approves
another test dependency.

## Dependency discipline

- Do not install or add a dependency without explaining why it is necessary.
- Keep the initial runtime dependencies limited to PyTorch and torchvision, with
  torchvision used only for the dataset and basic transforms.
- Do not commit `.venv`, downloaded data, or checkpoint files.

## Completion expectations

For each implementation step, report:

- files changed
- tensor shapes exercised
- tests or debug commands run
- any deviation from the reviewed plan
