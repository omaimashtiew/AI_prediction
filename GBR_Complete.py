import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mysql.connector
from datetime import datetime

# 1. الاتصال بقاعدة البيانات
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="omaima2003",
    database="tareeqy_db"
)

# 2. تحميل البيانات
data = pd.read_sql("SELECT * FROM tareeqy_fencestatus", conn)
conn.close()

# 3. معالجة البيانات
data['message_time'] = pd.to_datetime(data['message_time'])
data = data.sort_values(['fence_id', 'message_time'])

# حساب مدة الانتظار
data['next_time'] = data.groupby('fence_id')['message_time'].shift(-1)
data['duration'] = (data['next_time'] - data['message_time']).dt.total_seconds() / 60
data = data.dropna(subset=['duration'])
data = data[(data['duration'] > 0) & (data['duration'] < 720)]

# إنشاء الميزات
data['hour'] = data['message_time'].dt.hour
data['dayofweek'] = data['message_time'].dt.dayofweek
data['status_encoded'] = data['status'].map({'open':0, 'closed':1, 'sever_traffic_jam':2})

# 4. تدريب النموذج
features = ['fence_id', 'status_encoded', 'hour', 'dayofweek']
X = data[features]
y = data['duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['fence_id', 'status_encoded'])
    ],
    remainder='passthrough'
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# 5. التنبؤ لأحدث البيانات
latest_status = data.sort_values('message_time').drop_duplicates('fence_id', keep='last')
latest_status['predicted_wait'] = model.predict(latest_status[features]).round(1)

# 6. إنشاء HTML
html_content = f"""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>توقعات أوقات الانتظار - GBR</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; direction: rtl; }}
        h1 {{ color: #2c3e50; text-align: center; }}
        table {{ width: 80%; margin: 20px auto; border-collapse: collapse; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .footer {{ margin-top: 20px; text-align: center; font-size: 12px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>توقعات أوقات الانتظار على الحواجز</h1>
    <table>
        <tr>
            <th>رقم الحاجز</th>
            <th>الحالة</th>
            <th>الوقت المتوقع</th>
        </tr>
"""

for _, row in latest_status.iterrows():
    wait_time = row['predicted_wait']
    if wait_time > 60:
        hours = int(wait_time // 60)
        minutes = int(wait_time % 60)
        display_time = f"{hours} ساعة و {minutes} دقيقة"
    else:
        display_time = f"{wait_time} دقيقة"
    
    html_content += f"""
        <tr>
            <td>{row['fence_id']}</td>
            <td>{row['status']}</td>
            <td>{display_time}</td>
        </tr>
    """

html_content += f"""
    </table>
    <div class="footer">
        تم إنشاء التقرير في: {datetime.now().strftime("%Y-%m-%d %H:%M")} | نموذج Gradient Boosting
    </div>
</body>
</html>
"""

with open("GBR_Predictions.html", "w", encoding='utf-8') as f:
    f.write(html_content)

print("تم إنشاء التقرير بنجاح: GBR_Predictions.html")