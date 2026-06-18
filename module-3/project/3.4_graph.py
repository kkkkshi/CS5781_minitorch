import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.plot(
    [64, 128, 256, 512, 1024],
    [0.0719, 0.03688, 0.20240, 1.69526, 12.38390],
    "r",
    label="fast_ops",
)
plt.plot(
    [64, 128, 256, 512, 1024],
    [0.01053, 0.03380, 0.10478, 0.36983, 1.46291],
    "b",
    label="gpu",
)
plt.legend()
plt.xlabel("Size")
plt.ylabel("Time")
plt.show()
