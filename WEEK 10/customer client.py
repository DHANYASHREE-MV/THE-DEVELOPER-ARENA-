# Week 10: Client Project – Customer Segmentation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Sample mall customers dataset
data = {
    "CustomerID": range(1, 11),
    "Gender": ["Male","Female","Female","Male","Female","Male","Male","Female","Male","Female"],
    "Age": [19, 21, 20, 23, 31, 22, 35, 40, 18, 29],
    "Annual Income (k$)": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]
}
df = pd.DataFrame(data)

print("Sample Data:")
print(df.head())

# -------------------------------
# 2. Feature Selection & Scaling
# -------------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# -------------------------------
# 3. K-Means Clustering
# -------------------------------
# Elbow method
inertia = []
K = range(1, 8)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_X)
    inertia.append(model.inertia_)

plt.plot(K, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# Train final KMeans (choose k=3 for example)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_X)

print("\nClustered Customers:")
print(df[["CustomerID","Annual Income (k$)","Spending Score (1-100)","Cluster"]])

# -------------------------------
# 4. PCA for Visualization
# -------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_X)

plt.figure(figsize=(6,5))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df["Cluster"], cmap="viridis", s=80)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Customer Segments (via K-Means + PCA)")
plt.colorbar(label="Cluster")
plt.show()
