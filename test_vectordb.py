import os
from openai import OpenAI
import psycopg2
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "movies_db")
DB_USER = os.getenv("DB_USER", "manishsingh")
DB_PASSWORD = os.getenv("DB_PASSWORD", "manish123")

EMBEDDING_DIM = 1536

def get_top_k_context(question: str, k: int = 10):

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=question,
    )

    question_embedding = response.data[0].embedding
    embedding_string = f"[{','.join(map(str, question_embedding))}]"

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()

    search_sql = """
    SELECT
        movie_id,
        title,
        year,
        genres,
        plot_summary,
        actors,
        embedding <-> %s::vector AS distance
    FROM movie_embeddings
    ORDER BY embedding <-> %s::vector
    LIMIT %s;
    """

    cur.execute(search_sql, (embedding_string, embedding_string, k))
    results = cur.fetchall()

    cur.close()
    conn.close()
    return results

def build_prompt(question: str, top_results):
    """
    Builds a prompt that includes the user question plus the context from top_results.
    """
    context_texts = []
    for r in top_results:
        movie_id, title, year, genres, plot_summary, actors, distance = r
        similarity_score = 1 / (1 + distance)
        # Using a lower threshold to include more relevant matches
        if similarity_score > 0.3:  
            snippet = f"""
            Title: {title}
            Year: {year}
            Genres: {genres}
            Plot: {plot_summary}
            Actors: {actors}
            Similarity Score: {similarity_score:.3f}
            """
            context_texts.append(snippet.strip())

    context_block = "\n\n".join(context_texts)

    prompt = f"""
    You are a helpful movie recommendation assistant. Using ONLY the information provided below, answer the following question.
    For each movie you mention, include:
    1. Title and year
    2. Genres
    3. A brief explanation of why this movie is relevant to the question

    When recommending similar movies, consider:
    - Similar genres
    - Similar themes in the plot
    - Similar time period
    - Similar style or tone
    - Similar actors
    
    If a movie's similarity score is provided, use it to rank the recommendations, but focus more on explaining WHY the movies are similar.
    If you don't have enough relevant information, explain what aspects you were able to find matches for.

    Movie Information:
    {context_block}

    Question: {question}
    """
    return prompt

def ask_question(question: str, k: int = 10):
    """
    Ask a question about movies and get both the answer and the retrieved context.
    
    Args:
        question (str): The question to ask
        k (int): Number of similar movies to retrieve
        
    Returns:
        tuple: (answer, retrieved_movies) where answer is the LLM's response and
               retrieved_movies is a list of movie dictionaries
    """
    # 1. Retrieve top-k context rows
    rows = get_top_k_context(question, k=k)

    # 2. Build prompt for LLM
    prompt = build_prompt(question, rows)

    # 3. Call OpenAI with the final prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    # 4. Format retrieved movies
    retrieved_movies = []
    for row in rows:
        movie_id, title, year, genres, plot_summary, actors, distance = row
        retrieved_movies.append({
            "title": title,
            "year": year,
            "genres": genres,
            "plot_summary": plot_summary,
            "actors": actors,
            "similarity_score": 1 / (1 + distance)
        })

    return answer, retrieved_movies

if __name__ == "__main__":
    user_question = input("What is your question about the movies?")
    answer, movies = ask_question(user_question, k=10)
    print("----- LLM ANSWER -----")
    print(answer)