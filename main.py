import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import csv

points,clusterID = make_blobs(n_samples=70,n_features=2,centers=2)

with open("db/input.csv", mode="w", newline='') as myfile:
    writer = csv.writer(myfile)
    for x, y in points:  
        writer.writerow([x, y])

for ID in clusterID:
    print(ID)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(points[:,0], points[:,1])
plt.show()