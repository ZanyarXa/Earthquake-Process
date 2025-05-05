import pandas as pd
import sqlite3

conn = sqlite3.connect("earthquakes.db")

# ---------- 1. Magnitude vs Depth ----------
query1 = """
    SELECT magnitude, depth
    FROM earthquakes
    WHERE magnitude IS NOT NULL AND depth IS NOT NULL
    ORDER BY magnitude DESC
"""
df1 = pd.read_sql_query(query1, conn)

# ---------- 2. Earthquake Count by Source ----------
query2 = """
    SELECT source, COUNT(*) AS earthquake_count
    FROM earthquakes
    GROUP BY source
    ORDER BY earthquake_count DESC
"""
df2 = pd.read_sql_query(query2, conn)

# ---------- 3. Earthquake Count per Year ----------
query3 = """
    SELECT strftime('%Y', time) AS year, COUNT(*) AS earthquake_count
    FROM earthquakes
    WHERE time IS NOT NULL
    GROUP BY year
    ORDER BY year
"""
df3 = pd.read_sql_query(query3, conn)

# ---------- 4. Average Magnitude by Depth Category ----------
query4 = """
    SELECT 
        CASE 
            WHEN depth < 70 THEN 'Shallow'
            WHEN depth BETWEEN 70 AND 300 THEN 'Intermediate'
            ELSE 'Deep'
        END AS depth_category,
        AVG(magnitude) AS average_magnitude
    FROM earthquakes
    WHERE depth IS NOT NULL AND magnitude IS NOT NULL
    GROUP BY depth_category
    ORDER BY average_magnitude DESC
"""
df4 = pd.read_sql_query(query4, conn)


conn.close()

print("1. Magnitude vs Depth:\n", df1.head(), "\n")
print("2. Earthquake Count by Source:\n", df2, "\n")
print("3. Earthquake Count per Year:\n", df3, "\n")
print("4. Average Magnitude by Depth Category:\n", df4, "\n")

