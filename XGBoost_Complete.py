import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import os
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tabulate import tabulate

# إعدادات الملفات
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
    
    return df, le_status, le_city, le_day_part  # هذه السطر كان ناقصًا
  
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

            # تعريف wait_display بقيمة افتراضية
            wait_display = "0 دقيقة"
            predicted_wait = 0

            if current_status != 'open':
                hour = current_time.hour
                day_part = 'night' if hour < 6 else 'morning' if hour < 12 else 'afternoon' if hour < 18 else 'evening'
                
                # تأكد من معالجة القيم المفقودة
                fence_latitude = fence['latitude'] if pd.notna(fence['latitude']) else 0
                fence_longitude = fence['longitude'] if pd.notna(fence['longitude']) else 0
                fence_city = fence['city'] if pd.notna(fence['city']) else 'unknown'
                
                # تحقق من صحة البيانات قبل التحويل
                if current_status not in le_status.classes_:
                    status_encoded = 0
                else:
                    status_encoded = le_status.transform([current_status])[0]
                    
                if fence_city not in le_city.classes_:
                    city_encoded = 0
                else:
                    city_encoded = le_city.transform([fence_city])[0]
                
                if day_part not in le_day_part.classes_:
                    day_part_encoded = 0
                else:
                    day_part_encoded = le_day_part.transform([day_part])[0]
                
                # تحضير ميزات التنبؤ
                try:
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
                
                features_df = pd.DataFrame([features])
                features_scaled = scaler.transform(features_df)
                
                predicted_wait = int(round(model.predict(features_scaled)[0]))
                if current_status == 'closed':
                    predicted_wait = int(predicted_wait * 0.8)

                if predicted_wait >= 60:
                    hours = predicted_wait // 60
                    minutes = predicted_wait % 60
                    if minutes > 0:
                        wait_display = f"{hours} ساعة و {minutes} دقيقة"
                    else:
                        wait_display = f"{hours} ساعة"
                else:
                    wait_display = f"{predicted_wait} دقيقة"

        except Exception as e:
            print(f"خطأ في التنبؤ للبوابة {fence['id']}: {e}")
            predicted_wait = 15
            wait_display = "15 دقيقة"

        predictions.append({
            'fence_id': fence['id'],
            'fence_name': fence['name'],
            'city': fence['city'] if pd.notna(fence['city']) else 'غير معروف',
            'current_status': current_status,
            'predicted_wait': predicted_wait,
            'wait_display': wait_display
        })
    return predictions


# تعديل دالة load_or_train_model
def load_or_train_model(X, y):
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # قائمة النماذج التي نريد اختبارها
    regressors = {
        'XGBRegressor': xgb.XGBRegressor(
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
        ),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
    }
    
    # تقييم كل نموذج
    results = []
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': f"{mae:.2f}",
            'MedianAE': f"{medae:.2f}",
            'R2 Score': f"{r2:.4f}"
        })
    
    # عرض النتائج في جدول
    print("\n model comparassion  :")
    print(tabulate(results, headers="keys", tablefmt="pretty", showindex=False))
    
    # اختيار أفضل نموذج بناءً على R-squared
    best_model_info = max(results, key=lambda x: float(x['R2 Score']))
    best_model_name = best_model_info['Model']
    print(f"\n best model : {best_model_name}")
    print(f"with R-squared: {best_model_info['R2 Score']}")
    
    # استخدام أفضل نموذج
    best_model = regressors[best_model_name]
    best_model.fit(X_train, y_train)
    
    # حفظ النموذج
    joblib.dump(best_model, MODEL_PATH)
    
    return best_model

# تعديل دالة display_predictions لتعرض النتائج مباشرة
def main():
    try:
        # جلب البيانات
        fences_df, status_df = fetch_data()
        
        # التأكد من وجود بيانات كافية
        if fences_df.empty or status_df.empty:
            print("لا توجد بيانات كافية لبناء النموذج")
            return
        
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
        
        # عرض النتائج في جدول
        print("\n\nتوقعات أوقات الانتظار - تحديث:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        print(tabulate(predictions, headers="keys", tablefmt="pretty"))
        
    except Exception as e:
        print(f"حدث خطأ: {str(e)}")

if __name__ == "__main__":
    # إنشاء مجلد للنماذج إذا لم يكن موجوداً
    if not os.path.exists('models'):
        os.makedirs('models')
    
    main()


