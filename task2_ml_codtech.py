import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("sample_data_improved.csv")

# Clean column names to remove hidden spaces
df.columns = df.columns.str.strip()

# Feature selection (Input: quantity, Output: price)
X = df[["quantity"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R2):", r2_score(y_test, y_pred))

# Predict price for quantity = 5
test_input = pd.DataFrame([[5]], columns=["quantity"])
predicted_price = model.predict(test_input)[0]
print("Predicted price for quantity = 5:", predicted_price)
