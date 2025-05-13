import mysql.connector
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from flask import Flask, render_template, request
import joblib
import os
import math

# إنشاء تطبيق فلاسك
app = Flask(__name__)

# تكوين الاتصال بقاعدة البيانات
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="omaima2003",
        database="tareeqy_db"
    )
    return conn

# دالة لجلب البيانات من قاعدة البيانات
def fetch_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # استعلام للحصول على البيانات المدمجة من الجدولين
    query = """
    SELECT fs.id, fs.fence_id, f.name as checkpoint_name, fs.status, 
           fs.message_time, HOUR(fs.message_time) as hour_of_day, 
           DAYOFWEEK(fs.message_time) as day_of_week
    FROM tareeqy_fencestatus fs
    JOIN tareeqy_fence f ON fs.fence_id = f.id
    ORDER BY fs.fence_id, fs.message_time
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return pd.DataFrame(data)

# دالة لاستخراج ميزات الوقت
def extract_time_features(df):
    # تحويل عمود الوقت إلى تنسيق التاريخ
    df['message_time'] = pd.to_datetime(df['message_time'])
    
    # استخراج ميزات الوقت
    df['hour'] = df['message_time'].dt.hour
    df['day_of_week'] = df['message_time'].dt.dayofweek
    df['month'] = df['message_time'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 5 و 6 هما السبت والأحد
    
    return df

# دالة لحساب وقت الانتظار بين الحالات
def calculate_waiting_times(df):
    # ترتيب البيانات حسب الحاجز والوقت
    df = df.sort_values(['fence_id', 'message_time'])
    
    # إنشاء قاموس للاحتفاظ بآخر وقت للفتح لكل حاجز
    last_open_time = {}
    waiting_times = []
    
    for _, row in df.iterrows():
        fence_id = row['fence_id']
        current_time = row['message_time']
        status = row['status']
        
        # إذا كان الحاجز مغلقًا، احتفظ بالوقت
        if status == 'closed':
            last_open_time[fence_id] = None
            waiting_times.append(None)
        # إذا كان الحاجز مفتوحًا وكان مغلقًا في الماضي، احسب وقت الانتظار
        elif status == 'open':
            if fence_id in last_open_time and last_open_time[fence_id] is not None:
                # حساب وقت الانتظار بالدقائق
                waiting_time = (current_time - last_open_time[fence_id]).total_seconds() / 60
                waiting_times.append(waiting_time)
            else:
                waiting_times.append(None)
            last_open_time[fence_id] = current_time
        else:
            waiting_times.append(None)
    
    # إضافة وقت الانتظار إلى DataFrame
    df['waiting_time'] = waiting_times
    
    # حذف القيم الفارغة وضمان أن الوقت موجب
    df = df.dropna(subset=['waiting_time'])
    df['waiting_time'] = df['waiting_time'].apply(lambda x: abs(int(round(x))))
    
    return df

# دالة لتدريب نموذج التعلم الآلي
def train_model(df):
    # تجهيز البيانات للتدريب
    X = df[['hour', 'day_of_week', 'month', 'is_weekend']]
    
    # ترميز المتغيرات الفئوية
    categorical_features = ['fence_id', 'status']
    X_categorical = df[categorical_features]
    
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X_categorical)
    
    # دمج المتغيرات العددية والفئوية
    X_encoded = np.hstack((X.values, encoded_features))
    
    # متغير الهدف
    y = df['waiting_time'].values
    
    # تقسيم البيانات إلى مجموعات تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # تدريب نموذج Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # حفظ النموذج والترميز
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/checkpoint_model.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    
    return model, encoder

# دالة للتنبؤ بوقت الانتظار
def predict_waiting_time(fence_id, status, message_time):
    # تحميل النموذج والترميز
    try:
        model = joblib.load('models/checkpoint_model.pkl')
        encoder = joblib.load('models/encoder.pkl')
    except:
        # إذا لم يكن النموذج موجودًا، قم بإنشائه
        df = fetch_data()
        df = extract_time_features(df)
        df = calculate_waiting_times(df)
        model, encoder = train_model(df)
    
    # تحويل الوقت إلى تنسيق التاريخ
    if isinstance(message_time, str):
        message_time = datetime.strptime(message_time, '%Y-%m-%d %H:%M:%S')
    
    # استخراج ميزات الوقت
    hour = message_time.hour
    day_of_week = message_time.weekday()
    month = message_time.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # ترميز المتغيرات الفئوية
    categorical_data = pd.DataFrame({
        'fence_id': [fence_id],
        'status': [status]
    })
    encoded_categorical = encoder.transform(categorical_data)
    
    # إنشاء المصفوفة النهائية للتنبؤ
    X_predict = np.hstack(([[hour, day_of_week, month, is_weekend]], encoded_categorical))
    
    # التنبؤ
    predicted_time = model.predict(X_predict)[0]
    
    # التأكد من أن القيمة موجبة وتقريبها إلى أقرب عدد صحيح
    predicted_time = abs(int(round(predicted_time)))
    
    return predicted_time

# تهيئة النموذج
def initialize_model():
    if not os.path.exists('models/checkpoint_model.pkl'):
        try:
            df = fetch_data()
            df = extract_time_features(df)
            df = calculate_waiting_times(df)
            train_model(df)
            print("تم تدريب النموذج بنجاح وحفظه")
        except Exception as e:
            print(f"حدث خطأ أثناء تدريب النموذج: {str(e)}")

# الصفحة الرئيسية
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # جلب قائمة الحواجز
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name FROM tareeqy_fence")
        checkpoints = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"خطأ في الاتصال بقاعدة البيانات: {str(e)}")
        checkpoints = []
    
    prediction = None
    formatted_time = None
    error_message = None
    
    if request.method == 'POST':
        try:
            fence_id = int(request.form['checkpoint'])
            status = request.form['status']
            message_time = datetime.now()
            
            # التنبؤ بوقت الانتظار
            prediction = predict_waiting_time(fence_id, status, message_time)
            
            # تنسيق الوقت (أقل من 60 دقيقة بالدقائق، أكثر من 60 دقيقة بالساعات)
            if prediction < 60:
                formatted_time = f"{prediction} دقيقة"
            else:
                hours = prediction / 60
                formatted_time = f"{math.floor(hours)} ساعة و {prediction % 60} دقيقة"
        except Exception as e:
            error_message = f"حدث خطأ أثناء التنبؤ: {str(e)}"
    
    return render_template('index.html', 
                          checkpoints=checkpoints, 
                          prediction=prediction, 
                          formatted_time=formatted_time,
                          error_message=error_message)

# قالب HTML
@app.route('/templates/index.html')
def serve_template():
    return """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>التنبؤ بوقت الانتظار على الحواجز</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f9f9f9;
                text-align: right;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #555;
            }
            select, button {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
                margin-top: 10px;
            }
            button:hover {
                background-color: #2980b9;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background-color: #f0f7ff;
                border-radius: 8px;
                text-align: center;
            }
            .waiting-time {
                font-size: 24px;
                font-weight: bold;
                color: #e74c3c;
                margin: 10px 0;
            }
            .note {
                color: #7f8c8d;
                font-size: 14px;
                margin-top: 10px;
            }
            .status-open {
                color: green;
            }
            .status-closed {
                color: red;
            }
            .error {
                background-color: #ffebee;
                color: #c62828;
                padding: 10px;
                border-radius: 4px;
                margin-top: 20px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>التنبؤ بوقت الانتظار على الحواجز</h1>
            
            <form method="post">
                <div class="form-group">
                    <label for="checkpoint">اختر الحاجز:</label>
                    <select id="checkpoint" name="checkpoint" required>
                        {% for checkpoint in checkpoints %}
                            <option value="{{ checkpoint.id }}">{{ checkpoint.name }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="status">حالة الحاجز:</label>
                    <select id="status" name="status" required>
                        <option value="open" class="status-open">مفتوح</option>
                        <option value="closed" class="status-closed">مغلق</option>
                    </select>
                </div>
                
                <button type="submit">التنبؤ بوقت الانتظار</button>
            </form>
            
            {% if error_message %}
            <div class="error">
                {{ error_message }}
            </div>
            {% endif %}
            
            {% if prediction %}
            <div class="result">
                <h2>نتيجة التنبؤ</h2>
                <p>وقت الانتظار المتوقع:</p>
                <div class="waiting-time">{{ formatted_time }}</div>
                <p class="note">ملاحظة: هذا التنبؤ مبني على البيانات السابقة وقد يختلف عن الوقت الفعلي.</p>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

# دالة للتهيئة عند بدء التطبيق
def create_folders():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # كتابة قالب HTML إلى ملف
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(serve_template())
    
    # تهيئة النموذج
    initialize_model()

if __name__ == '__main__':
    # تنفيذ دالة التهيئة قبل تشغيل التطبيق
    create_folders()
    app.run(debug=True)