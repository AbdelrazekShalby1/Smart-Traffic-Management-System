import sqlite3

DB_PATH = "DataBase/traffic_system.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Delete all violations
cursor.execute("DELETE FROM violations")

deleted = cursor.rowcount

conn.commit()
conn.close()

print(f"Deleted records: {deleted}")
print("All database data deleted successfully.")