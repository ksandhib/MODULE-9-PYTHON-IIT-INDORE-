import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Task 1: Load customer data
df = pd.read_csv('customers.csv')
print("Data loaded successfully.")
print("First 5 rows:")
print(df.head())

# Task 2: Check the number of cells in each column with null values
null_counts = df.isnull().sum()
print("\nNull values per column:")
print(null_counts)

# Task 3: Create a scatter plot with Age as X and Spending Score as Y
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter Plot of Age vs Spending Score')
plt.savefig('scatter_age_spending.png')  # Save plot to file
plt.close()
print("\nScatter plot saved as 'scatter_age_spending.png'")

# Task 4: Perform K-Means Clustering and draw scatter plot colored by clusters
# Using Age and Spending Score for clustering
X = df[['Age', 'Spending Score (1-100)']]

# Assuming K=5 for this dataset (common choice)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Scatter plot with clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clusters (K=5) based on Age and Spending Score')
plt.legend()
plt.savefig('scatter_clusters.png')  # Save plot to file
plt.close()
print("Clustered scatter plot saved as 'scatter_clusters.png'")

# Final output: Centroids and cluster counts
print("\nCluster Centroids (Age, Spending Score):")
centroids = kmeans.cluster_centers_
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1}: Age={centroid[0]:.2f}, Spending Score={centroid[1]:.2f}")

print("\nNumber of customers in each cluster:")
cluster_counts = df['Cluster'].value_counts().sort_index()
for i in range(5):
    print(f"Cluster {i+1}: {cluster_counts[i]} customers")
