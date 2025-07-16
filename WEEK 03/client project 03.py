import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset("iris")

# 1. Scatter Plot: Sepal Length vs Sepal Width
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.title("Sepal Length vs Sepal Width")
plt.show()

# 2. Histogram: Petal Length
sns.histplot(df["petal_length"], bins=15, kde=True, color="green")
plt.title("Distribution of Petal Length")
plt.show()

# 3. Boxplot: Petal Width by Species
sns.boxplot(x="species", y="petal_width", data=df)
plt.title("Petal Width by Species")
plt.show()

# 4. Heatmap: Feature Correlation
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Iris Feature Correlation")
plt.show()

