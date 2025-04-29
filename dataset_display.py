from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Show the first 5 rows
print(df.head())


# Plot sepal length vs sepal width
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.ylabel('Sepal width (cm)')
plt.xlabel('Sepal length (cm)')
plt.title('Iris Dataset - Sepal Length vs Width')
plt.show()