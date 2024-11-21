from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset with make_circles
X, y = make_circles(n_samples=1000, noise=0.05)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a scatterplot of the training data
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train)

# Set the title and show the plot
plt.title("Train Data")
plt.show()
