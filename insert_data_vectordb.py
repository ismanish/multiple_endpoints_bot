from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import json
import os
load_dotenv()



client = OpenAI()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "movies_db")
DB_USER = os.getenv("DB_USER", "manishsingh")
DB_PASSWORD = os.getenv("DB_PASSWORD", "manish123")

EMBEDDING_DIM = 1536

def create_table_and_insert_data(json_file_path: str, batch_size: int = 20):

    conn=psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD)
    conn.autocommit = False  # We'll manage transactions manually for batching
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Drop existing table if it exists
    cur.execute("DROP TABLE IF EXISTS movie_embeddings;")
    conn.commit()

    # Create fresh table
    create_table_sql = f"""
    CREATE TABLE movie_embeddings (
        id SERIAL PRIMARY KEY,
        movie_id TEXT,
        title TEXT,
        year TEXT,
        genres TEXT,
        plot_summary TEXT,
        actors TEXT,
        embedding vector({EMBEDDING_DIM})
    );
    """
    cur.execute(create_table_sql)
    conn.commit()

    print("Created fresh movie_embeddings table")
    # Read JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    # Process in batches
    batch=[]
    movie_texts = []

    for idx, movie in enumerate(movies):
        movie_id = movie.get("movie_id","")
        title = movie.get("title","")
        year = movie.get("year","")
        genres = movie.get("genres",[])
        plot_summary = movie.get("plot_summary","")
        actors = movie.get("actors",[])

        movie_text = f"Title: {title}. Year: {year}. Genres: {genres}. Plot: {plot_summary}. Starring: {actors}"
        batch.append((movie_id, title, year, genres, plot_summary, actors))
        movie_texts.append(movie_text)

        if len(batch) == batch_size or idx == len(movies) - 1:
            try:
                response = client.embeddings.create(model="text-embedding-ada-002", input=movie_texts)
                embeddings = [item.embedding for item in response.data]
                # Prepare batch insert values
                values = []
                for (movie_id, title, year, genres, plot_summary, actors), embedding in zip(batch, embeddings):
                    values.append((movie_id, title, year, genres, plot_summary, actors, embedding))
                                # Batch insert using executemany
                insert_sql = """
                INSERT INTO movie_embeddings 
                    (movie_id, title, year, genres, plot_summary, actors, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                cur.executemany(insert_sql, values)
                conn.commit()
                print(f"Inserted {len(batch)} movies into movie_embeddings table")
                batch = []
                movie_texts = []
            
            except Exception as e:
                print(f"Error inserting batch: {e}")
                conn.rollback()

    cur.close()
    conn.close()

    print("Finished inserting data into movie_embeddings table")

if __name__ == "__main__":
    JSON_FILE_PATH = "movie_data.json"  # Update if needed
    create_table_and_insert_data(JSON_FILE_PATH)
