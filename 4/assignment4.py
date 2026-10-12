import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Task 1: Create np array from the data
data = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91]])

print("Data array:")
print(data)

# Task 2: Generate scatter plot
plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1])
plt.title('Scatter Plot of Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('scatter_plot.png')
plt.close()  # Close to avoid display issues

print("Scatter plot saved as 'scatter_plot.png'")

# Task 3: Plot dendrogram
linked = linkage(data, 'single')  # Using single linkage for agglomerative clustering
plt.figure(figsize=(10,7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Agglomerative Clustering')
plt.savefig('dendrogram.png')
plt.close()

print("Dendrogram saved as 'dendrogram.png'")
