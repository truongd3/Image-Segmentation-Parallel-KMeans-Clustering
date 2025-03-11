import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import csv

points, clusterID = make_blobs(n_samples=70, n_features=2, centers=2)

with open("db/input.csv", mode="w", newline='') as myfile:
    writer = csv.writer(myfile)
    for x, y in points:  
        writer.writerow([x, y])

test_root = "tests"
os.makedirs(test_root, exist_ok=True)

existing_tests = [d for d in os.listdir(test_root) if d[0:4] == "test" and d[4:].isdigit()]
if existing_tests:
    max_index = max(int(d[4:]) for d in existing_tests)
    new_test_folder = f"test{max_index + 1}"
else:
    new_test_folder = "test1"

new_test_path = os.path.join(test_root, new_test_folder)
os.makedirs(new_test_path)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(points[:,0], points[:,1])
image_path = os.path.join(new_test_path, f"input{new_test_folder[4:]}.jpg")
plt.savefig(image_path)
# plt.show()

# Save cluster assignments to a text file
output_path = os.path.join(new_test_path, f"output{new_test_folder[4:]}_expected.txt")
with open(output_path, "w") as f:
    for i in range(len(points)):
        f.write(f"Point ({points[i][0]:.5f}, {points[i][1]:.5f}) -> Cluster {clusterID[i]}\n")