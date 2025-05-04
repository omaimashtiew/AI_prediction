# predict.py
import joblib
import pandas as pd

model = joblib.load("wait_time_predictor.pkl")

# مثال: حاجز رقم 7، الساعة 17، يوم الاثنين
example = pd.DataFrame([{
    'fence_id': 7,
    'latitude': 32.2227,
    'longitude': 35.2621,
    'hour': 17,
    'dayofweek': 0,
    'is_weekend': 0
}])

prediction = model.predict(example)[0]
wait_time = max(0, round(prediction))
print(f"التوقع: ستنتظر حوالي {wait_time} دقيقة عند الحاجز")
