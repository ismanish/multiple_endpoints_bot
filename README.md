# Multi-Endpoint Movie Chatbot

A sophisticated chatbot system that leverages both SQL and RAG (Retrieval-Augmented Generation) capabilities to provide comprehensive information about movies, including rental data, plot summaries, and detailed movie information.
## Please Ref https://github.com/ismanish/text_sql_final for the SQL Endpoint
File: sql_endpoint.py

## 🌟 Key Features

1. **Dual Database Integration**
   - SQL Database (Pagila): Handles rental statistics and basic movie information
   - Vector Database: Manages detailed plot summaries and movie content

2. **Smart Query Routing**
   - Automatically determines whether to use SQL, RAG, or both based on query type
   - Optimizes response accuracy by choosing the most appropriate data source

3. **Rich Information Retrieval**
   - Basic movie details (release year, rating, language)
   - Rental statistics and popularity metrics
   - Detailed plot summaries and themes
   - Actor and character information

4. **Interactive Interface**
   - Web-based GUI using Gradio
   - Clean and intuitive user interface
   - Persistent chat history

## 🛠️ Technical Architecture

The system consists of several core components:

### Core Components
- `chatbot.py`: Main chatbot implementation with conversation management and response generation
- `pipeline_orchestration.py`: Handles query routing and orchestrates the flow between SQL and RAG endpoints
- `rag_endpoint.py`: Manages RAG-based queries using vector embeddings
- `gradio_app.py`: Implements the web interface using Gradio

### Data Management
- `prepare_cmu_movie_data.py`: Processes and prepares movie data for the vector database
- `insert_data_vectordb.py`: Handles the insertion of processed data into the vector database
- `test_vectordb.py`: Contains tests for vector database functionality

### Supporting Files
- `requirements.txt`: Lists all Python dependencies
- `.env.example`: Template for environment variables configuration

## 📋 Prerequisites

```
python >= 3.8
openai
langchain
chromadb
gradio
psycopg2-binary
python-dotenv
```

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key
   - Configure your PostgreSQL database connection string

4. Prepare the databases:
   ```bash
   # Prepare movie data
   python prepare_cmu_movie_data.py
   
   # Initialize vector database
   python insert_data_vectordb.py
   ```

5. Start the services:
   ```bash
   # Start RAG endpoint
   python rag_endpoint.py
   
   # Start web interface
   python gradio_app.py
   ```

## 💡 Usage Examples

1. **Basic Movie Information (SQL)**
   ```
   "What are the top 5 most rented comedy movies?"
   "Which actors appear most frequently in our rental inventory?"
   ```

2. **Plot and Content Queries (RAG)**
   ```
   "Tell me about movies involving time travel"
   "What are some movies similar to Inception?"
   ```

3. **Combined Queries (SQL + RAG)**
   ```
   "Find horror movies with high rental counts and describe their plots"
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for language model support
- Pagila database for rental statistics
