from typing import Annotated, Dict, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
import requests
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Define state
class State(TypedDict):
    messages: List[BaseMessage]
    next_step: str
    current_response: str
    error: str | None

# Tool functions
def call_rag_endpoint(question: str) -> Dict:
    response = requests.post(
        "http://localhost:8000/rag",
        json={"text": question, "k": 5}
    )
    return response.json()

def call_sql_endpoint(question: str) -> Dict:
    response = requests.post(
        "http://localhost:8001/sql",
        json={"text": question}
    )
    return response.json()

def determine_route(state: State) -> State:
    """Determine which endpoint(s) to use based on the question."""
    last_message = state["messages"][-1].content
    
    llm = ChatOpenAI(temperature=0)
    prompt = f"""Given the user question: '{last_message}'
    We have two different movie databases:

    1. Pagila (SQL): Contains basic movie information and rental data:
       - Basic info: title, release year, rating, language, duration
       - Rental data: rental counts, inventory, popularity metrics
       - Example queries: 
         - "When was X released?"
         - "What's the rating of X?"
         - "most rented movies"
         - "rental counts"
         - "popular genres"
    
    2. CMU Movie Summaries (RAG): Contains detailed movie information:
       - Plot summaries
       - Themes and story elements
       - Actor roles and character information
       - Example queries:
         - "What is the plot of X?"
         - "movies about Y theme"
         - "who played Z character?"
         - "find movies similar to X"

    Determine if this question requires:
    1. SQL (Pagila) - for rental/inventory/basic stats
    2. RAG (CMU Summaries) - for plot/content/detailed info
    3. BOTH - only if the question specifically needs to combine rental data with detailed movie information
    
    Respond with only one of: 'RAG', 'SQL', or 'BOTH'
    
    Question type analysis:
    1. If asking about rentals, counts, or popularity -> SQL
    2. If asking about plot, story, or movie details -> RAG
    3. If explicitly combining rental metrics with movie details -> BOTH
    
    If user has specifically asked to ue SQL then use the 'SQL' or 'BOTH'
    if you think the question asked is based on previous question and if previous question used 'SQL'
    then just use 'SQL"""
    
    response = llm.invoke(prompt).content.strip().upper()
    state["next_step"] = response
    return state

def route_to_node(state: State) -> str:
    """Route to the appropriate node based on the next_step."""
    if state["next_step"] == "RAG":
        return "rag_node"
    elif state["next_step"] == "SQL":
        return "sql_node"
    return "both_node"

def rag_node(state: State) -> State:
    """Handle RAG-specific queries."""
    question = state["messages"][-1].content
    try:
        result = call_rag_endpoint(question)
        state["current_response"] = result["answer"]
    except Exception as e:
        state["error"] = str(e)
        state["current_response"] = f"Error calling RAG endpoint: {str(e)}"
    return state

def sql_node(state: State) -> State:
    """Handle SQL-specific queries."""
    question = state["messages"][-1].content
    try:
        result = call_sql_endpoint(question)
        state["current_response"] = f"SQL Query: {result['sql_query']}\nAnswer: {result['answer']}"
    except Exception as e:
        state["error"] = str(e)
        state["current_response"] = f"Error calling SQL endpoint: {str(e)}"
    return state

def both_node(state: State) -> State:
    """Handle queries requiring both RAG and SQL."""
    question = state["messages"][-1].content
    try:
        # Get responses from both endpoints independently
        rag_result = call_rag_endpoint(question)
        sql_result = call_sql_endpoint(question)
        
        # Use LLM to integrate the responses
        llm = ChatOpenAI(temperature=0)
        integration_prompt = f"""Given a user question and two different sources of movie information, create a coherent, integrated response.

User Question: {question}

Source 1 (Database/SQL Information about rentals and statistics):
{sql_result['answer']}

Source 2 (Movie Details/Plot Information):
{rag_result['answer']}

Please analyze both sources and create a comprehensive response that:
1. Integrates relevant information from both sources
2. Highlights any interesting connections or patterns
3. Addresses the user's question completely
4. Acknowledges if certain information is missing from either source

Response should be clear, well-organized, and natural-sounding."""

        integrated_response = llm.invoke(integration_prompt).content
        state["current_response"] = integrated_response
        
    except Exception as e:
        state["error"] = str(e)
        state["current_response"] = f"Error processing combined query: {str(e)}"
    return state

def create_workflow() -> StateGraph:
    """Create the workflow graph."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("determine_route", determine_route)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("both_node", both_node)
    
    # Set entry point
    workflow.set_entry_point("determine_route")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "determine_route",
        route_to_node,
        {
            "rag_node": "rag_node",
            "sql_node": "sql_node",
            "both_node": "both_node"
        }
    )
    
    # Add edges to END
    workflow.add_edge("rag_node", END)
    workflow.add_edge("sql_node", END)
    workflow.add_edge("both_node", END)
    
    return workflow.compile()

def process_question(question: str) -> Dict:
    """Process a question through the workflow."""
    workflow = create_workflow()
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "next_step": "",
        "current_response": "",
        "error": None
    }
    
    final_state = workflow.invoke(initial_state)
    
    return {
        "question": question,
        "response": final_state["current_response"],
        "route_taken": final_state["next_step"],
        "error": final_state["error"]
    }

if __name__ == "__main__":
    # Test the orchestrator
    questions = [
        "What are the top 5 most rented comedy movies?",  # SQL (rental data from Pagila)
        "Tell me about movies involving time travel and their plots",  # RAG (plot info from CMU)
        "Find horror movies with high rental counts and describe their plots",  # BOTH (combines rental data with plot info)
        "Which actors appear most frequently in our rental inventory?",  # SQL
        "What are some movies with similar plot to Carry On Jatta",  # RAG
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = process_question(question)
        print(f"Route: {result['route_taken']}")
        print(f"Response: {result['response']}")
        if result['error']:
            print(f"Error: {result['error']}")