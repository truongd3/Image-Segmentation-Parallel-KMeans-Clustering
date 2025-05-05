import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("db/report/performance_by_pixelsize_hybrid.csv")

plt.figure()
plt.plot(df["Pixels Size"], df["Segmentation Time"], marker="o", linestyle="-", color="#264f73ff",
         markerfacecolor='#3fbfb2ff', markeredgecolor='#3fbfb2ff')

for x, y in zip(df['Pixels Size'], df['Segmentation Time']):
    plt.text(
        x, y,            # position at the point
        f'{y:.3f}',      # text string
        ha='center',     # horizontal alignment
        va='bottom',     # vertical alignment just above
        fontsize=10       # tweak size as needed
    )

plt.xlabel("Pixels Size")
plt.ylabel("Segmentation Time (s)")
plt.title("K-Means Segmentation Time vs. Image Size")
plt.grid(True)
plt.tight_layout()
plt.show()
