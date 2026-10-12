from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load iris data
iris = load_iris()

# Step 2: Extract the data property
data = iris.data

# Step 3: Train an Agglomerative Clustering model
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
model.fit(data)

# Print cluster labels as text output
print("Cluster labels:", model.labels_)

# Step 4: Plot dendrogram to visualize the clustering linkage
Z = linkage(data, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Agglomerative Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')  # Save the plot as PNG
plt.show()  # This will display the plot if running interactively
