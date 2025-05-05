import csv
import matplotlib.pyplot as plt

# 1) Read the single row of times from the CSV
with open('db/report/dog48_k=5.csv', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    row = next(reader)
    times = [float(x) for x in row]

# 2) Define labels for each version
versions = ['Plain', 'MPI', 'CUDA', 'MPI+CUDA']

# 3) Plot
plt.figure()
bars = plt.bar(versions, times, color='#264f73ff')
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{height:.3f}',
        ha='center', va='bottom'
    )

plt.xlabel('Implementation')
plt.ylabel('Segmentation Time (s)')
plt.title('dog48.jpg k=5 Segmentation Time by Version')
plt.tight_layout()
plt.show()
