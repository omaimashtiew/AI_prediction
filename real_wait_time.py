import pandas as pd
import mysql.connector
from datetime import timedelta

# معلومات الاتصال
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="omaima2003",
    database="tareeqy_db",
    pool_size=5
)

# الكلمات المفتاحية
STATUS_KEYWORDS = {
    "open": ["open","✅","مفتوح", "مفتوحة", "سالك", "سالكة","نظيف","فتحت","سالكه","فاتحات","فتح"],
    "closed": ["closed","🔴","⛔️","❌","مغلق", "مغلقة", "سكر" ,"مسكر","مغلقه"," وقوف تام"," واقف"],
    "sever_traffic_jam": ["sever_traffic_jam","ازمة", "مازم", "كثافة سير","ازمه","حاجز","مخصوم","🛑"],
}

# دالة لتصنيف الحالة
def classify_status(s):
    for status_type, keywords in STATUS_KEYWORDS.items():
        for word in keywords:
            if word in s:
                return status_type
    return "unknown"

# جلب البيانات
query = """
SELECT fence_id, status, message_time
FROM tareeqy_fencestatus
ORDER BY fence_id, message_time
"""
df = pd.read_sql(query, conn)

# المعالجة
df["status_type"] = df["status"].apply(classify_status)
df["message_time"] = pd.to_datetime(df["message_time"])

# حساب زمن الانتظار
records = []

for fence_id, group in df.groupby("fence_id"):
    group = group.sort_values("message_time").reset_index(drop=True)
    for i in range(1, len(group)):
        prev_status = group.loc[i - 1, "status_type"]
        curr_status = group.loc[i, "status_type"]

        prev_time = group.loc[i - 1, "message_time"]
        curr_time = group.loc[i, "message_time"]
        delta = curr_time - prev_time

        # شرط أن الفارق الزمني لا يتجاوز 24 ساعة
        if delta <= timedelta(hours=24):
            if (prev_status in ["closed", "sever_traffic_jam"]) and curr_status == "open":
                records.append({
                    "fence_id": fence_id,
                    "status_from": prev_status,
                    "status_to": curr_status,
                    "start_time": prev_time,
                    "end_time": curr_time,
                    "real_wait_time": round(delta.total_seconds() / 60, 1)  # بالدقائق
                })
            elif prev_status == "closed" and curr_status == "sever_traffic_jam":
                records.append({
                    "fence_id": fence_id,
                    "status_from": prev_status,
                    "status_to": curr_status,
                    "start_time": prev_time,
                    "end_time": curr_time,
                    "real_wait_time": round(delta.total_seconds() / 60, 1)
                })

# تحويل النتيجة إلى DataFrame
wait_df = pd.DataFrame(records)

# حفظها إذا حبيت
wait_df.to_csv("estimated_real_wait_times.csv", index=False)


# متوسط وزمن الوسيط لكل حاجز
stats_per_fence = wait_df.groupby("fence_id")["real_wait_time"].agg(["mean", "median", "count"]).reset_index()
print(stats_per_fence)



