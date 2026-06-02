import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lattice import adjacency_list

data = np.load("ising.npz")

plt.figure(figsize=(10, 5))

plt.plot(data["energy"], label="Demixed")

plt.xlabel("Steps")
plt.ylabel("Demixed Parameter")
plt.legend()

plt.savefig("ising.png")