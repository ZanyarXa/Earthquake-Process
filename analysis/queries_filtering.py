
import pandas as pd
import sqlite3

conn = sqlite3.connect("earthquakes.db")


df1 = pd.read_sql_query("""
    SELECT * FROM earthquakes
    WHERE magnitude > 6
    ORDER BY magnitude DESC
    LIMIT 10
""", conn)

df2 = pd.read_sql_query("""
    SELECT 
        CASE
            WHEN magnitude < 5 THEN 'Less than 5'
            WHEN magnitude >= 5 AND magnitude < 6 THEN '5 to 6'
            WHEN magnitude >= 6 AND magnitude < 7 THEN '6 to 7'
            WHEN magnitude >= 7 AND magnitude < 8 THEN '7 to 8'
            ELSE '8 and above'
        END AS magnitude_range,
        COUNT(*) AS count
    FROM earthquakes
    GROUP BY magnitude_range
    ORDER BY magnitude_range
""", conn)


df3 = pd.read_sql_query("""
    SELECT id, time, magnitude, depth, type
    FROM earthquakes
    WHERE magnitude > 6
    ORDER BY depth DESC
    LIMIT 20
""", conn)


df4 = pd.read_sql_query("""
    SELECT location_source, COUNT(*) AS earthquake_count
    FROM earthquakes
    GROUP BY location_source
    ORDER BY earthquake_count DESC
    LIMIT 20
""", conn)


conn.close()

print("Top 10 Earthquakes with Magnitude > 6")
print(df1, "\n")

print("Earthquake Count by Magnitude Range")
print(df2, "\n")

print("Top 20 Deep Earthquakes with Magnitude > 6")
print(df3, "\n")

print("Top 20 Location Sources by Earthquake Count")

