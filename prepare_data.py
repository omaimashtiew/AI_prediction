import pandas as pd

status = pd.read_csv("fencestatus.csv")
fences = pd.read_csv("fences.csv")

# دمج الجداول
df = status.merge(fences, left_on="fence_id", right_on="id", suffixes=('', '_fence'))

# ترتيب حسب الحاجز والوقت
df['message_time'] = pd.to_datetime(df['message_time'])
df = df.sort_values(by=['fence_id', 'message_time']).reset_index(drop=True)

# نحفظ وقت الانتظار المحسوب
df['real_wait_minutes'] = None

# نمر على كل حاجز لحاله
for fence_id, group in df.groupby('fence_id'):
    last_congested_time = None
    for i, row in group.iterrows():
        status = row['status']
        time = row['message_time']

        if status in ['sever_traffic_jam', 'closed']:
            last_congested_time = time

        elif status == 'open' and last_congested_time:
            wait = (time - last_congested_time).total_seconds() / 60  # بالدقائق
            df.at[i, 'real_wait_minutes'] = wait
            last_congested_time = None  # نبدأ من جديد

# حذف الصفوف اللي ما إلها وقت انتظار محسوب
df = df[df['real_wait_minutes'].notna()]

# خصائص زمنية
df['hour'] = df['message_time'].dt.hour
df['dayofweek'] = df['message_time'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# الأعمدة المهمة
features = df[['fence_id', 'latitude', 'longitude', 'hour', 'dayofweek', 'is_weekend', 'real_wait_minutes']]
features.to_csv("processed_data.csv", index=False)
