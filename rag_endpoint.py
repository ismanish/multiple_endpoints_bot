from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from test_vectordb import ask_question

app = FastAPI(
    title="CMU Movie RAG API",
    description="API for querying movie information using RAG (Retrieval Augmented Generation)",
    version="1.0.0"
)

class Question(BaseModel):
    text: str
    k: Optional[int] = 10

class MovieInfo(BaseModel):
    title: Optional[str] = ""
    year: Optional[str] = ""
    genres: Optional[str] = ""
    plot_summary: Optional[str] = ""
    actors: Optional[str] = ""
    similarity_score: Optional[float] = None

# class Answer(BaseModel):
#     question: str
#     answer: str
#     retrieved_movies: List[MovieInfo]

class Answer(BaseModel):
    answer: str

@app.post("/rag", response_model=Answer)
async def rag(question: Question):
    try:
        # Get answer and retrieved movies using the modified ask_question function
        answer, retrieved_movies = ask_question(question.text, k=question.k)
        
        # Ensure all fields are strings, replace None with empty string
        cleaned_movies = []
        for movie in retrieved_movies:
            cleaned_movie = {
                "title": str(movie.get("title", "")) if movie.get("title") is not None else "",
                "year": str(movie.get("year", "")) if movie.get("year") is not None else "",
                "genres": str(movie.get("genres", "")) if movie.get("genres") is not None else "",
                "plot_summary": str(movie.get("plot_summary", "")) if movie.get("plot_summary") is not None else "",
                "actors": str(movie.get("actors", "")) if movie.get("actors") is not None else "",
                "similarity_score": movie.get("similarity_score")
            }
            cleaned_movies.append(cleaned_movie)
        
        # return {
        #     "question": question.text,
        #     "answer": answer,
        #     "retrieved_movies": cleaned_movies
        # }
        return {
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("rag_endpoint:app", host="0.0.0.0", port=8000, reload=True)
