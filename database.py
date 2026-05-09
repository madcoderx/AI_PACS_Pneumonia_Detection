import sqlite3

conn = sqlite3.connect('pacs.db')

cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    prediction TEXT
)
''')

conn.commit()

conn.close()

print("Database created successfully")