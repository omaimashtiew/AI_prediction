import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import os
from flask import Flask, render_template_string
import matplotlib.pyplot as plt

app = Flask(__name__)
MODEL_PATH = "wait_time_model.pkl"
SCALER_PATH = "scaler.pkl"
CLUSTER_PATH = "kmeans_model.pkl"

# تحسين اتصال قاعدة البيانات مع إعادة المحاولة
def get_db_connection(max_retries=3):
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host="yamabiko.proxy.rlwy.net",
                port=26213,
                user="root",
                password="sbKIFwBCaymbcggetPSaFpblUvThYNSX",
                database="railway",
                charset='utf8mb4',
                connect_timeout=5
            )
            return conn
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2)

# جلب البيانات مع ذاكرة التخزين المؤقت
def fetch_data(cache_file='data_cache.pkl'):
    if os.path.exists(cache_file):
        df = joblib.load(cache_file)
        return df['fences'], df['status']
    
    conn = get_db_connection()
    try:
        fences_df = pd.read_sql("SELECT id, name, latitude, longitude, city FROM tareeqy_fence", conn)
        status_df = pd.read_sql("""
            SELECT fs.id, fs.fence_id, fs.status, fs.message_time 
            FROM tareeqy_fencestatus fs
            WHERE fs.message_time >= NOW() - INTERVAL 6 MONTH
        """, conn)
        
        joblib.dump({'fences': fences_df, 'status': status_df}, cache_file)
        return fences_df, status_df
    finally:
        conn.close()

# حساب الميزات الجغرافية
def calculate_geo_features(fences_df):
    fences_df['geo_cluster'] = 0
    if len(fences_df) > 1:
        coords = fences_df[['latitude', 'longitude']].dropna().values
        
        # تأكد من وجود بيانات كافية للتجميع
        if len(coords) > 3:  # نحتاج على الأقل 3 نقاط للتجميع
            distortions = []
            K = range(1, min(10, len(coords)))
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(coords)
                distortions.append(kmeans.inertia_)
            
            optimal_k = 3
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            
            # تعيين قيم التجميع للصفوف التي تحتوي على إحداثيات غير مفقودة فقط
            mask = fences_df[['latitude', 'longitude']].notna().all(axis=1)
            fences_df.loc[mask, 'geo_cluster'] = kmeans.fit_predict(fences_df.loc[mask, ['latitude', 'longitude']].values)
            joblib.dump(kmeans, CLUSTER_PATH)
    
    return fences_df

# معالجة البيانات
def preprocess_data(fences_df, status_df):
    # معالجة القيم المفقودة في fences_df
    fences_df['city'].fillna('unknown', inplace=True)
    
    # الإحداثيات المفقودة - استخدام قيم وسيطة للمدينة نفسها
    unique_cities = fences_df['city'].unique()
    for city in unique_cities:
        city_data = fences_df[fences_df['city'] == city]
        
        # حساب الوسيط للقيم غير المفقودة
        lat_median = city_data['latitude'].dropna().median()
        lon_median = city_data['longitude'].dropna().median()
        
        # استخدام الوسيط فقط إذا كان هناك قيم صالحة
        if not pd.isna(lat_median):
            fences_df.loc[(fences_df['city'] == city) & (fences_df['latitude'].isna()), 'latitude'] = lat_median
        if not pd.isna(lon_median):
            fences_df.loc[(fences_df['city'] == city) & (fences_df['longitude'].isna()), 'longitude'] = lon_median
    
    # استبدال أي قيم مفقودة متبقية بمتوسط عام
    fences_df['latitude'].fillna(fences_df['latitude'].dropna().mean(), inplace=True)
    fences_df['longitude'].fillna(fences_df['longitude'].dropna().mean(), inplace=True)
    
    # حساب الميزات الجغرافية
    fences_df = calculate_geo_features(fences_df)
    
    # دمج البيانات
    df = pd.merge(status_df, fences_df, left_on='fence_id', right_on='id', how='left', suffixes=('_status', '_fence'))
    
    # تأكد من أن حقل message_time هو تاريخ/وقت
    df['message_time'] = pd.to_datetime(df['message_time'])
    
    # ميزات زمنية
    df['hour'] = df['message_time'].dt.hour
    df['day_of_week'] = df['message_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['message_time'].dt.month
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)
    df['day_part'] = pd.cut(df['hour'], 
                          bins=[0, 6, 12, 18, 24],
                          labels=['night', 'morning', 'afternoon', 'evening'])
    
    # ترميز الميزات
    le_status = LabelEncoder()
    le_city = LabelEncoder()
    le_day_part = LabelEncoder()
    
    df['status_encoded'] = le_status.fit_transform(df['status'])
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['day_part_encoded'] = le_day_part.fit_transform(df['day_part'])
    
    # حساب زمن الانتظار
    def calculate_wait_time(group):
        group = group.sort_values()  # تأكد من الترتيب الزمني الصحيح
        diffs = group.diff().dt.total_seconds()
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 0:
            return positive_diffs.median() / 60  # تحويل من ثوان إلى دقائق
        return 10  # قيمة افتراضية معقولة
    
    # تطبيق حساب وقت الانتظار لكل مجموعة
    wait_times = df.groupby(['fence_id', 'status', 'hour', 'day_of_week'])['message_time'].apply(calculate_wait_time)
    
    # دمج أوقات الانتظار مع الإطار الأصلي
    wait_times = wait_times.reset_index(name='wait_time')
    df = pd.merge(df, wait_times, on=['fence_id', 'status', 'hour', 'day_of_week'], how='left')
    
    # معالجة القيم المفقودة في wait_time
    df['wait_time'].fillna(10, inplace=True)  # قيمة افتراضية
    
    # تقليم القيم المتطرفة
    df['wait_time'] = df['wait_time'].clip(lower=5, upper=120)
    
    return df, le_status, le_city, le_day_part

# تحضير بيانات التدريب
def prepare_training_data(df):
    features = [
        'fence_id', 'latitude', 'longitude', 'hour', 'day_of_week',
        'is_weekend', 'month', 'status_encoded', 'city_encoded',
        'geo_cluster', 'is_rush_hour', 'day_part_encoded'
    ]
    
    # تأكد من وجود جميع الأعمدة
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"العمود {feature} غير موجود في البيانات")
    
    X = df[features]
    y = df['wait_time']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    
    return X_scaled, y

# تدريب النموذج - تصحيح مشكلة Early Stopping
# تدريب النموذج - النسخة المعدلة
def load_or_train_model(X, y):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            if hasattr(model, 'n_features_in_') and model.n_features_in_ != X.shape[1]:
                print(f"تحذير: عدم تطابق الميزات! سيتم حذف النموذج القديم. متوقع {X.shape[1]} ميزات، وجدنا {model.n_features_in_}")
                os.remove(MODEL_PATH)
                raise FileNotFoundError
            return model
        except Exception as e:
            print(f"خطأ في تحميل النموذج: {e}. سيتم إعادة التدريب...")
    
    # تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # تعريف النموذج مع معلمات معقولة
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # تدريب النموذج بدون early stopping
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.2f} دقيقة")
    print(f"Median Absolute Error: {medae:.2f} دقيقة")
    print(f"R2 Score: {r2:.2f}")
    
    # حفظ النموذج
    joblib.dump(model, MODEL_PATH)
    
    
    return model

# التنبؤ
def predict_all_fences(model, fences_df, current_time, le_status, le_city, le_day_part):
    try:
        # تحميل النماذج المساعدة
        scaler = joblib.load(SCALER_PATH)
        
        # محاولة تحميل نموذج التجميع، إذا فشل استخدم نموذج افتراضي
        try:
            kmeans = joblib.load(CLUSTER_PATH)
        except:
            print("لم يتم العثور على نموذج التجميع، سيتم إنشاء نموذج افتراضي")
            # تدريب نموذج تجميع افتراضي على البيانات المتاحة
            coords = fences_df[['latitude', 'longitude']].dropna().values
            if len(coords) > 2:
                kmeans = KMeans(n_clusters=min(3, len(coords)), random_state=42)
                kmeans.fit(coords)
            else:
                # إذا لم تكن هناك بيانات كافية، استخدم دالة تنبؤ تعيد 0 دائمًا
                class DummyModel:
                    def predict(self, X):
                        return np.zeros(len(X))
                kmeans = DummyModel()
            joblib.dump(kmeans, CLUSTER_PATH)
    except Exception as e:
        print(f"خطأ في تحميل النماذج المساعدة: {e}")
        # إنشاء نماذج افتراضية
        scaler = StandardScaler()
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
        kmeans = DummyModel()
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(kmeans, CLUSTER_PATH)

    predictions = []
    
    for _, fence in fences_df.iterrows():
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM tareeqy_fencestatus WHERE fence_id = %s ORDER BY message_time DESC LIMIT 1",
                (fence['id'],)
            )
            result = cursor.fetchone()
            current_status = result[0] if result else 'unknown'
            conn.close()

            if current_status == 'open':
                predicted_wait = 0
            else:
                hour = current_time.hour
                day_part = 'night' if hour < 6 else 'morning' if hour < 12 else 'afternoon' if hour < 18 else 'evening'
                
                # تأكد من معالجة القيم المفقودة
                fence_latitude = fence['latitude'] if pd.notna(fence['latitude']) else 0
                fence_longitude = fence['longitude'] if pd.notna(fence['longitude']) else 0
                fence_city = fence['city'] if pd.notna(fence['city']) else 'unknown'
                
                # تحقق من صحة البيانات قبل التحويل
                if current_status not in le_status.classes_:
                    status_encoded = 0  # قيمة افتراضية
                else:
                    status_encoded = le_status.transform([current_status])[0]
                    
                if fence_city not in le_city.classes_:
                    city_encoded = 0  # قيمة افتراضية
                else:
                    city_encoded = le_city.transform([fence_city])[0]
                
                if day_part not in le_day_part.classes_:
                    day_part_encoded = 0  # قيمة افتراضية
                else:
                    day_part_encoded = le_day_part.transform([day_part])[0]
                
                # تحضير ميزات التنبؤ
                try:
                    # حساب التجميع الجغرافي
                    if pd.isna(fence_latitude) or pd.isna(fence_longitude):
                        geo_cluster = 0
                    else:
                        geo_cluster = kmeans.predict([[fence_latitude, fence_longitude]])[0]
                except:
                    geo_cluster = 0
                
                features = {
                    'fence_id': fence['id'],
                    'latitude': fence_latitude,
                    'longitude': fence_longitude,
                    'hour': hour,
                    'day_of_week': current_time.weekday(),
                    'is_weekend': 1 if current_time.weekday() in [5, 6] else 0,
                    'month': current_time.month,
                    'status_encoded': status_encoded,
                    'city_encoded': city_encoded,
                    'geo_cluster': geo_cluster,
                    'is_rush_hour': 1 if (7 <= hour <= 10) or (14 <= hour <= 16) else 0,
                    'day_part_encoded': day_part_encoded
                }
                
                # تحويل الميزات إلى DataFrame
                features_df = pd.DataFrame([features])
                
                # تطبيع القيم
                features_scaled = scaler.transform(features_df)
                
                # التنبؤ
                predicted_wait = model.predict(features_scaled)[0]
                
                # تعديل التنبؤ بناءً على الحالة الحالية
                if current_status == 'sever_traffic_jam':
                    predicted_wait = min(predicted_wait * 1.8, 100)
                elif current_status == 'closed':
                    predicted_wait = min(predicted_wait * 1.09, 90)
                else:
                    predicted_wait = min(predicted_wait, 60)
                
                # تقريب التنبؤ للحصول على أرقام أكثر واقعية
        except Exception as e:
            print(f"خطأ في التنبؤ للبوابة {fence['id']}: {e}")
            predicted_wait = 15  # قيمة افتراضية في حالة الخطأ

        predictions.append({
            'fence_id': fence['id'],
            'fence_name': fence['name'],
            'city': fence['city'] if pd.notna(fence['city']) else 'غير معروف',
            'current_status': current_status,
            'predicted_wait': predicted_wait
        })
    return predictions

HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl">
<head>
    <title>توقعات أوقات الانتظار</title>
    <meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 0 auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        th, td { padding: 12px; text-align: right; border: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .low-wait { background-color: #d4edda; }
        .medium-wait { background-color: #fff3cd; }
        .high-wait { background-color: #f8d7da; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .status-info { margin-top: 20px; text-align: center; }
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            border-radius: 5px;
            font-weight: bold;
        }
        .open { background-color: #d4edda; color: #155724; }
        .moderate { background-color: #fff3cd; color: #856404; }
        .closed { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>توقعات أوقات الانتظار - تحديث: {{ current_time }}</h1>
        <div class="status-info">
            <span class="status-badge open">وقت انتظار منخفض (أقل من 15 دقيقة)</span>
            <span class="status-badge moderate">وقت انتظار متوسط (15-30 دقيقة)</span>
            <span class="status-badge closed">وقت انتظار طويل (أكثر من 30 دقيقة)</span>
        </div>
        <table>
            <tr>
                <th>البوابة</th>
                <th>المدينة</th>
                <th>الحالة الحالية</th>
                <th>وقت الانتظار المتوقع (دقيقة)</th>
            </tr>
            {% for pred in predictions %}
            <tr class="{% if pred.predicted_wait <= 15 %}low-wait
                      {% elif pred.predicted_wait <= 30 %}medium-wait
                      {% else %}high-wait{% endif %}">
                <td>{{ pred.fence_name }}</td>
                <td>{{ pred.city }}</td>
                <td>{{ pred.current_status }}</td>
                <td>{{ pred.predicted_wait }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

@app.route('/')
def display_predictions():
    try:
        # جلب البيانات
        fences_df, status_df = fetch_data()
        
        # التأكد من وجود بيانات كافية
        if fences_df.empty or status_df.empty:
            return "لا توجد بيانات كافية لبناء النموذج", 500
        
        # معالجة البيانات
        df, le_status, le_city, le_day_part = preprocess_data(fences_df, status_df)
        
        # تحضير بيانات التدريب
        X, y = prepare_training_data(df)
        
        # تدريب النموذج أو تحميله
        model = load_or_train_model(X, y)
        
        # إجراء التنبؤ
        current_time = datetime.now()
        predictions = predict_all_fences(model, fences_df, current_time, le_status, le_city, le_day_part)
        
        # فرز البوابات حسب وقت الانتظار (تنازلياً)
        predictions = sorted(predictions, key=lambda x: x['predicted_wait'], reverse=True)
        
        # عرض الصفحة مع التنبؤات
        return render_template_string(HTML_TEMPLATE, 
                                  predictions=predictions, 
                                  current_time=current_time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"حدث خطأ: {str(e)}<br><pre>{error_details}</pre>", 500

if __name__ == "__main__":
    # إنشاء مجلد للنماذج إذا لم يكن موجوداً
    if not os.path.exists('models'):
        os.makedirs('models')
    
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))