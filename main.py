# linear-regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([5, 7, 9, 11, 13, 15])

# Split data is here 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model coefficient (slope):", model.coef_[0])
print("Model intercept:", model.intercept_)
print("Model score:", model.score(X_test, y_test))

# Prediction
y_pred = model.predict(X)

# Plot results will come here
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, label="Predicted", linewidth=2)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression Example")
plt.legend()
plt.show()

# Saveing the model  model
joblib.dump(model, "linear_regression_model.pkl")
print("Model saved as linear_regression_model.pkl")
