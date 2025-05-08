import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("db/report/multiproc_performance.csv")

plt.figure()
plt.plot(df["Number of Processes"], df["Segmentation Time of MPI"], marker="o", linestyle="-", label="MPI", color="#3fbfb2ff", markerfacecolor='#3fbfb2ff', markeredgecolor="#3fbfb2ff")
plt.plot(df["Number of Processes"], df["Segmentation Time of MPI & CUDA"], marker="s", linestyle="--", label="MPI + CUDA", color="#264f73ff", markerfacecolor='#264f73ff', markeredgecolor="#264f73ff")

plt.xlabel("Number of Processes")
plt.ylabel("Segmentation Time (s)")
plt.title("MPI vs MPI + CUDA Segmentation Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()