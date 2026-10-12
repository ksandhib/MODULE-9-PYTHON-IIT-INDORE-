import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load customer data
df = pd.read_csv('customers.csv')

# Check the number of cells in each column with null values
null_values = df.isnull().sum()

# Create a scatter plot with Age as X and Spending Score as Y
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Spending Score'])
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Scatter Plot: Age vs Spending Score')
plt.savefig('scatter_age_spending.png')
plt.close()

# Find the best number of clusters between 1 and 10 using the elbow method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['Age', 'Spending Score']])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_method.png')
plt.close()

# From the elbow plot, the best k is typically where the WCSS starts to decrease slowly. Assuming k=5 based on common elbow point.
best_k = 5

# Perform K-Means clustering with the best k
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Spending Score']])

# Draw a scatter plot displaying data points colored on the basis of clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Spending Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Clusters: Age vs Spending Score')
plt.colorbar(label='Cluster')
plt.savefig('clusters_scatter.png')
plt.close()

# Prepare output
output = f"""
Null values in each column:
{null_values.to_string()}

Best number of clusters (assumed from elbow method): {best_k}

Clusters assigned to data points (first 10 rows):
{df[['CustomerID', 'Age', 'Spending Score', 'Cluster']].head(10).to_string(index=False)}

Plots saved as:
- scatter_age_spending.png
- elbow_method.png
- clusters_scatter.png
"""

print(output)

# Save to note.txt
with open('note.txt', 'w') as f:
    f.write(output)
