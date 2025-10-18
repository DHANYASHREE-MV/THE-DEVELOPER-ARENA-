# Hands-On EDA using Titanic Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')

# Dataset shape & info
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation analysis (numeric only)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()

# Visualizations
sns.countplot(x='sex', data=df)
plt.title("Gender Distribution")
plt.show()

sns.histplot(df['age'].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='pclass', y='age', data=df)
plt.title("Passenger Class vs Age")
plt.show()

sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival by Gender")
plt.show()


























