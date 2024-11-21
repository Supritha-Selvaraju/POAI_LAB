import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Example dataset
years_of_experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
salary = np.array([40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])

# Prepare the data
X = years_of_experience.reshape(-1, 1)  # Features (years of experience)
y = salary  # Target (salary)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the salaries for the test set
y_pred = model.predict(X_test)

# Plotting the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

r_squared = model.score(X_test, y_test)
print(f'R-squared: {r_squared:.2f}')

# Predict salary for 6 years of experience
predicted_salary = model.predict([[6]])
print(f'Predicted salary for 6 years of experience: {predicted_salary[0]:.2f}')

# Print the model's coefficients
print(f'Slope (Coefficient): {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')
