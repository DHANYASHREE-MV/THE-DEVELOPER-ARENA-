# Week 10: Clustering & Dimensionality Reduction
# Hands-On: K-Means + PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# -------------------------------
# 1. Generate Sample Customer Data
# -------------------------------
# (If you don’t have a dataset, we simulate one)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=["Annual Income", "Spending Score"])

print("Sample of dataset:")
print(df.head())

# -------------------------------
# 2. Standardize Data
# -------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# -------------------------------
# 3. Apply K-Means
# -------------------------------
# Choosing optimal k using Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    inertia.append(model.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Train final model with k=4 (example)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

print("\nCluster counts:")
print(df['Cluster'].value_counts())

# -------------------------------
# 4. Dimensionality Reduction with PCA
# -------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot clusters in 2D PCA space
plt.figure(figsize=(6,5))
plt.scatter(pca_data[:,0], pca_data[:,1], c=clusters, cmap="viridis", s=50)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Customer Clusters (via K-Means + PCA)")
plt.colorbar(label="Cluster")
plt.show()
