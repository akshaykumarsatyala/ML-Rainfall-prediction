import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("rainfall.csv")

# Keep only required columns
# CHANGE 'Temperature' if your column name is different
data = data[['Temperature', 'Rainfall']]

# Handle missing values
data = data.fillna(data.mean())

# One input feature (Temperature)
X = data[['Temperature']]   # 1 feature
y = data['Rainfall']        # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained using ONLY Temperature")
