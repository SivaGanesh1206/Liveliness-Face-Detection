


from sklearn.neighbors import KNeighborsClassifier
import logging

logging.basicConfig(level=logging.INFO)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Setup logger
logger = logging.getLogger(__name__)


def predict_face_identity(new_embedding, embeddings, labels,mobile_number, threshold=0.61):
    """
    Predict the identity of a face using cosine similarity with a threshold.
    """
    if len(embeddings) == 0:
        return "Unknown"

    # Compute cosine similarity
    new_embeddings = new_embedding / np.linalg.norm(new_embedding)
    similarities = cosine_similarity(new_embeddings.reshape(1, -1), embeddings)
    max_similarity_index = np.argmax(similarities)
    max_similarity_score = similarities[0][max_similarity_index]
    
    if max_similarity_score >= threshold:
        mobile_number = mobile_number[max_similarity_index]
        # return labels[max_similarity_index]

        return [labels[max_similarity_index],max_similarity_score,max_similarity_index,mobile_number]
    else:
        return ["unknown",max_similarity_score,max_similarity_index,mobile_number]

        # return "Unknown"




def calculate_mean_embedding(embeddings):
    """
    Calculate the mean of multiple embeddings to create a single representative embedding.
    """
    mean_embedding = np.mean(embeddings, axis=0)  # Compute mean
    return mean_embedding / np.linalg.norm(mean_embedding)  # Compute the mean across embeddings



import psycopg2
import numpy as np
from psycopg2.pool import SimpleConnectionPool

# Connection pool with min 1 and max 10 connections
db_pool = SimpleConnectionPool(1, 10, dbname='dbname', user='username', password='*****', host='localhost', port='portnumber')

def save_embeddings_to_db(embedding, label, user):
    if not isinstance(embedding, np.ndarray):
        print("Invalid embedding: Not a NumPy array")
        return

    print(f"Attempting to save embedding for user: {user} with label: {label}")

    conn = db_pool.getconn()
    try:
        with conn.cursor() as curs:
            query = "INSERT INTO tablename (embedding, mobile_number, user_name) VALUES (%s, %s, %s)"
            values = (embedding.tolist(), label, user)
            
            print(f"Executing query: {query}")
            curs.execute(query, values)

        conn.commit()
        print("Embedding successfully saved to database.")
    except Exception as e:
        print(f"Error saving embedding: {e}")
    finally:
        db_pool.putconn(conn)
        print("Database connection returned to pool.")



def load_embeddings_from_db(db_config):
    """
    Load embeddings and labels from a PostgreSQL database.
    """
    embeddings = []
    labels = []
    mobile_number=[]
    
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Fetch embeddings and labels
        cursor.execute("SELECT user_name,mobile_number, embedding FROM embeddings")
        rows = cursor.fetchall()
        
        for row in rows:
            label = row[0]
            mobile=row[1]
            raw_embedding = np.array(row[2])
            
            try:
                embedding = np.array(raw_embedding)
                if embedding.shape != (512,):  # or whatever your expected shape is
                    print(f"Skipping malformed embedding for {label}, shape: {embedding.shape}")
                    continue  # Convert to numpy array
                labels.append(label)
                mobile_number.append(mobile)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Failed to convert embedding for {label}: {e}")
                continue
    except Exception as e:
        print(f"Error loading embeddings: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    return np.array(embeddings) , np.array(labels),np.array(mobile_number)



