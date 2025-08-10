# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the Penguins dataset
df = sns.load_dataset('penguins')

# Set the visual theme
sns.set(style="whitegrid")

#  1. Scatter Plot: Bill Length vs Depth (relationship)
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title("Bill Length vs Bill Depth by Species")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.tight_layout()
plt.show()

#  2. Histogram: Flipper Length Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='flipper_length_mm', bins=20, kde=True, color='teal')
plt.title("Distribution of Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.tight_layout()
plt.show()

#  3. Boxplot: Body Mass by Species
plt.figure(figsize=(6, 4))
sns.boxplot(x='species', y='body_mass_g', data=df, palette="pastel")
plt.title("Body Mass Comparison by Species")
plt.xlabel("Species")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.show()

#  4. Heatmap: Correlation Matrix
plt.figure(figsize=(6, 4))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ✅ 5. Pairplot (optional - shows multiple relationships together)
sns.pairplot(df.dropna(), hue='species')
plt.suptitle("Pairwise Relationships in Penguin Features", y=1.02)
plt.show()
