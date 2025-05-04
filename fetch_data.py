# fetch_data.py
import mysql.connector
import pandas as pd

# الاتصال بالقاعدة
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="omaima2003",
    database="tareeqy_db"
)

# نسحب البيانات
fences = pd.read_sql("SELECT * FROM tareeqy_fence", conn)
status = pd.read_sql("SELECT * FROM tareeqy_fencestatus", conn)

conn.close()

# حفظهم كـ CSV مؤقت
fences.to_csv("fences.csv", index=False)
status.to_csv("fencestatus.csv", index=False)
