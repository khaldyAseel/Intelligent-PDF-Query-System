import sqlite3

# Connect to the SQLite database
db_path = "text_database.db"  # Ensure the correct path if the DB is in another folder
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if the "documents" table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
table_exists = cursor.fetchone()

if table_exists:
    print("✅ Table 'documents' exists in the database.")

    # Count total records in the table
    cursor.execute("SELECT COUNT(*) FROM documents;")
    total_records = cursor.fetchone()[0]
    print(f"📌 Total documents stored: {total_records}")

    if total_records > 0:
        # Retrieve the first 5 documents for preview
        cursor.execute("SELECT id, content FROM documents LIMIT 5;")
        rows = cursor.fetchall()
        print("\n🔍 Sample Data from 'documents' Table:")
        for row in rows:
            print(f"🆔 ID: {row[0]}\n📄 Content: {row[1][:300]}...\n{'-'*50}")

else:
    print("❌ Table 'documents' does NOT exist. You may need to recreate it.")

# Close the connection
conn.close()