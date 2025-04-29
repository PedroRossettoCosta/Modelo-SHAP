from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
housing = fetch_california_housing()

# Create a DataFrame with proper housing labels
df = pd.DataFrame(housing.data, columns=[
    'MedInc',         # Median income in block group
    'HouseAge',       # Median house age in block group
    'AveRooms',       # Average number of rooms per household
    'AveBedrms',      # Average number of bedrooms per household
    'Population',     # Block group population
    'AveOccup',       # Average number of household members
    'Latitude',       # Block group latitude
    'Longitude'       # Block group longitude
])
df['Price'] = housing.target * 100000  # Convert target to actual dollar values

# Show the first 5 rows with meaningful columns
print("California Housing Dataset Sample:")
print(df.head())

# Plot income vs house age colored by price
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['MedInc'], df['HouseAge'], c=df['Price'],cmap='viridis', alpha=0.6, edgecolors='w')

plt.colorbar(scatter, label='Home Price ($)')
plt.xlabel('Median Household Income ($100,000)')
plt.ylabel('Median House Age (Years)')
plt.title('California Housing: Income vs Age vs Price')
plt.grid(True, alpha=0.3)

# Format ticks for better readability
plt.gca().xaxis.set_major_formatter('${x:1.1f}')
plt.xticks(ticks=np.arange(0, 16, 2), labels=np.arange(0, 16, 2)/10)

plt.show()