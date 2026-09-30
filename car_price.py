import pandas as pd

# Load dataset
df = pd.read_csv("car_data.csv.csv")

# Select input features
X = df[
    ["Year", "Present_Price", "Driven_kms",
     "Fuel_Type", "Selling_type", "Transmission", "Owner"]
]

# Select target
y = df["Selling_Price"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Categorical columns
categorical = ["Fuel_Type", "Selling_type", "Transmission"]

# Convert text columns into numbers
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"
)

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()