import psycopg2
import numpy as np

def create_database():
    conn = psycopg2.connect(
            dbname="dbname", user="username", password="*****", host="localhost", port="port"
        )
    conn.autocommit = True  # Needed to execute CREATE DATABASE command
    cursor = conn.cursor()

    # Step 2: Create the 'embeddings' database if it doesn't exist
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'dbname';")
    if not cursor.fetchone():
        cursor.execute('CREATE DATABASE dbname;')
        print("Database 'dbname' created.")
    else:   
        print("Database 'dbname' already exists.")
    """
    Create the database table for storing embeddings if they don't exist.
    """
    # conn = psycopg2.connect(
    #     dbname="embeddings", u    ser="testuser", password="test@123", host="localhost",port="5432"
    # )
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dbname (
            id SERIAL PRIMARY KEY,  
            user_name TEXT NOT NULL,
            mobile_number VARCHAR,
            embedding FLOAT8[][] NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def update_embedding_for_user(embedding, mobile_number):
    import psycopg2
    import numpy as np

    db_config = {
        'dbname': 'dbname',
        'user': 'username',
        'password': '****',
        'host': 'localhost',
        'port': 'portnumber'
    }

    """Update embedding for the existing user based on mobile_number (empcode)."""

    try:
        conn = psycopg2.connect(**db_config)
        curs = conn.cursor()
        print("Connected to the database.")

        if not isinstance(embedding, np.ndarray):
            print("Error: Embedding must be a numpy array.")
            return

        embedding_list = embedding.tolist()

        # Convert mobile_number to string (in case it's passed as int)
        mobile_number_str = str(mobile_number)

        # Check if mobile number (empcode) exists
        curs.execute('SELECT 1 FROM tablename WHERE primarykey_column = %s LIMIT 1', (mobile_number_str,))
        if curs.fetchone():
            # Update the embedding column
            curs.execute('''
                UPDATE tablename
                SET embedding = %s
                WHERE primarykey_column = %s
            ''', (embedding_list, mobile_number_str))
            print(f"Updated embedding for mobile number (primary_key): {mobile_number_str}")
        else:
            print(f"Mobile number {mobile_number_str} not found. Cannot update embedding.")
            return {"status": "error", "message": f"User with primarykey {mobile_number_str} not found."}

        conn.commit()
        print("Embedding successfully updated.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if curs:
            curs.close()
        if conn:
            conn.close()
        print("Database connection closed.")






def save_embeddings_to_db(embedding, label,user):
    db_config = {
    'dbname': 'dbname',
    'user': 'username',
    'password': '*****',
    'host': 'localhost',  
    'port': 'portnumber'        
}
    """Save embeddings and their corresponding labels to a PostgreSQL database."""
    
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(**db_config)
        curs = conn.cursor()
        print("Connected to the database.")

        # Iterate through embeddings and labels
        if not isinstance(embedding, np.ndarray):
            print("Error: Embedding must be a numpy array.")
            return

        # for label, embedding in zip(labels, embeddings):
        #     print(f"Embedding value {embedding}")
        # Convert the embedding numpy array to a list (PostgreSQL can store lists)
        embedding_list = embedding.tolist()

        # Insert the data into the database
        curs.execute('''
                 INSERT INTO tablename (embedding, mobile_number, user_name) VALUES (%s, %s, %s)
             ''', (embedding_list, label, user))
            
        print(f"Inserted embedding for label: {label}")

    # Commit the transaction and close the connection
        conn.commit()
        print("Embeddings saved to the database.")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
         # Close the cursor and connection
        if curs:
            curs.close()
        if conn:
            conn.close()
        print("Database connection closed.")


# def save_embeddings_to_db(embedding, label):
#     db_config = {
#     'dbname': 'face_embeddings',
#     'user': 'neural-nexus',
#     'password': 'neural@123',
#     'host': 'localhost',  
#     'port': '5432'        
# }
#     """Save embeddings and their corresponding labels to a PostgreSQL database."""
    
#     # Connect to PostgreSQL
#     try:
#         conn = psycopg2.connect(**db_config)
#         cursor = conn.cursor()
#         print("Connected to the database.")

#         # Iterate through embeddings and labels
#         if not isinstance(embedding, np.ndarray):
#             print("Error: Embedding must be a numpy array.")
#             return
#         embedding_list = embedding.tolist()

#         # Insert the data into the database
#         cursor.execute(
#             "INSERT INTO embeddings (label, embedding, user_name) VALUES (%s, %s, %s)",
#             (label, embedding_list, None)
#         )
#         print(f"Saved embedding for label: {label}")

#     # Commit the transaction and close the connection
#         conn.commit()
#         print("Embeddings saved to the database.")
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         # Close the cursor and connection
#         if cursor:
#             cursor.close()
#         if conn:
#             conn.close()



def load_embeddings_from_db(db_config):
    """
    Load embeddings and labels from a PostgreSQL database.
    """
    embeddings = []
    labels = []
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Fetch embeddings and labels
        cursor.execute("SELECT user_name, embedding FROM embeddings")
        rows = cursor.fetchall()
        
        for row in rows:
            label = row[0]
            embedding = np.array(row[1])  # Convert to numpy array
            labels.append(label)
            embeddings.append(embedding)
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return np.array(embeddings), np.array(labels)


