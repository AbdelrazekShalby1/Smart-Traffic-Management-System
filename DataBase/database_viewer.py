import sqlite3
import pandas as pd


DB_PATH = "DataBase/traffic_system.db"


# Connect to database
conn = sqlite3.connect(DB_PATH)


# Read violations
query = """
SELECT
    id,
    vehicle_id,
    vehicle_type,
    violation_type,
    speed,
    license_plate,
    timestamp,
    evidence_path
FROM violations
ORDER BY vehicle_id DESC
"""


df = pd.read_sql_query(query, conn)


conn.close()


# Display
print("\n==========================================")
print("        TRAFFIC VIOLATIONS")
print("==========================================\n")


if df.empty:

    print("No violations found.")

else:

    print(df.to_string(index=False))


    print("\n==========================================")
    print("Total Violations:", len(df))
    print("==========================================")

    print(
        "\nViolation Types:"
    )

    print(
        df["violation_type"].value_counts()
    )