from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

print("Loading California housing data...")
housing = fetch_california_housing()

print("\nDataset info:")
print(f"Number of samples: {housing.data.shape[0]}")
print(f"Number of features: {housing.data.shape[1]}")
print(f"Feature names: {housing.feature_names}")

print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Model 1: Linear Regression
print("\n--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"MSE: {lr_mse:.4f}")
print(f"R2 Score: {lr_r2:.4f}")

# Model 2: Random Forest
print("\n--- Random Forest ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print(f"MSE: {rf_mse:.4f}")
print(f"R2 Score: {rf_r2:.4f}")

# Save the better model
if rf_r2 > lr_r2:
    print("\nRandom Forest performed better! Saving it...")
    with open('best_housing_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("Random Forest model saved!")
else:
    print("\nLinear Regression performed better! Saving it...")
    with open('best_housing_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print("Linear Regression model saved!")

print("\nDone!")