# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("processed_data.csv")

X = df.drop("estimated_wait", axis=1)
y = df["estimated_wait"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

# تقييم مبدأي
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"MAE = {mae:.2f} دقيقة")

# حفظ الموديل
import joblib
joblib.dump(model, "wait_time_predictor.pkl")
