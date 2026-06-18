# CS5781 MiniTorch

My coursework for CS5781 (Machine Learning Engineering, Cornell Tech). Across five
modules it builds a small PyTorch-style library from scratch — operators, scalar
autodiff, tensors, fast CPU/GPU kernels, and finally the layers needed to train
real conv nets. Each module reuses the code from the one before it.

## Modules

| Folder | What it covers |
|--------|----------------|
| [`module-0`](module-0) | Elementary operators and the `Module` / `Parameter` base classes |
| [`module-1`](module-1) | Scalar autodifferentiation: a computation graph and backprop |
| [`module-2`](module-2) | Tensors: storage, strides, broadcasting, and tensor autodiff |
| [`module-3`](module-3) | Speed: parallel CPU kernels (Numba) and CUDA GPU kernels |
| [`module-4`](module-4) | Neural networks: convolution, pooling, softmax, and CNNs |

Each module's own README covers what was implemented there. `git-fundamentals` is
the warm-up Git exercise; `extra credit` holds the puzzler notebooks.

## Running

Each module is self-contained. From inside one:

```bash
pip install -r requirements.txt
pytest                 # all tests; or e.g. `pytest -m task3_1` for one task
```

Module 4 also needs MNIST data — `pip install python-mnist`, then run
`mnist_get_data.sh`. The downloaded data and the per-module virtualenvs are
gitignored.

## Notes

Shared files (`operators.py`, `module.py`, `tensor_*.py`, …) are copied forward
between modules with `sync_previous_module.py`, so each module keeps its own full
copy instead of importing from a sibling.

Reference docs: https://minitorch.github.io/
