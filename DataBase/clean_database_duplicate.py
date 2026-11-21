import sqlite3

DB_PATH = "traffic_system.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Delete duplicate violations
# Keep the first record for each vehicle + violation type
cursor.execute("""
DELETE FROM violations
WHERE id NOT IN (
    SELECT MIN(id)
    FROM violations
    GROUP BY vehicle_id, violation_type
)
""")

deleted = cursor.rowcount

conn.commit()
conn.close()

print(f"Deleted duplicates: {deleted}")
print("Database cleaned successfully.")