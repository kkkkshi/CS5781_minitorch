# MiniTorch Module 4

Neural networks: convolution, pooling, and the layers needed to train real image
and text classifiers.

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/
- Overview: https://minitorch.github.io/module4.html

## What's implemented

- **`minitorch/fast_conv.py`** — `conv1d` and `conv2d`, forward and backward, as
  Numba-JIT kernels written as a direct index-math accumulation. A `reverse`
  flag turns the same loop into the transposed convolution the backward pass
  needs.
- **`minitorch/nn.py`**
  - `tile` + `avgpool2d` / `maxpool2d` — pooling by folding each window into the
    last dimension and reducing it.
  - `Max` (with an argmax-based gradient), `softmax`, and a numerically stable
    `logsoftmax` (log-sum-exp with the row max subtracted out).
  - `dropout`, with an `ignore` flag so it becomes a no-op at eval time.
- **`project/run_sentiment.py`** — a Kim-style sentiment CNN: three parallel 1D
  convolutions (widths 3/4/5) + ReLU, max-over-time, a linear layer, dropout,
  and a sigmoid.
- **`project/run_mnist_multiclass.py`** — a LeNet-style MNIST classifier: two 2D
  conv + ReLU layers, average pooling, two linear layers with dropout, and a
  logsoftmax over the 10 classes.

## Setup

Download the MNIST data (needs `wget`, e.g. `brew install wget` on Mac):

```
pip install python-mnist
mnist_get_data.sh
```

## Tests

```
python run_tests.py
```

## Building on Module 3

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```
