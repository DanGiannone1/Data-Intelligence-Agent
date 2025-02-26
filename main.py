#main.py

import os
import json
import asyncio
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Azure SDK and LLM imports
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Type utilities
from typing import List, Dict, Any, Union, Literal, TypedDict, Set

# Import the Service Bus handler and the prompt
from service_bus_handler import ServiceBusHandler
from prompts import query_prompt

# Load environment variables
load_dotenv()

# --- Configuration & Setup ---
ai_search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
ai_search_key = os.environ["AZURE_SEARCH_KEY"]
ai_search_index = os.environ["AZURE_SEARCH_INDEX"]

aoai_deployment = os.getenv("AOAI_DEPLOYMENT")
aoai_key = os.getenv("AOAI_KEY")
aoai_endpoint = os.getenv("AOAI_ENDPOINT")

search_client = SearchClient(ai_search_endpoint, ai_search_index, AzureKeyCredential(ai_search_key))

MAX_ATTEMPTS = 3
NUM_SEARCH_RESULTS = 5
K_NEAREST_NEIGHBORS = 30

# --- Type Definitions ---
SearchResultIndex = Literal[tuple(range(NUM_SEARCH_RESULTS))]

class SearchResult(TypedDict):
    id: str
    content: str
    source_file: str
    source_pages: int
    score: float

class ReviewDecision(BaseModel):
    thought_process: str
    valid_results: List[SearchResultIndex]
    invalid_results: List[SearchResultIndex]
    decision: Literal["retry", "finalize"]

class SearchPromptResponse(BaseModel):
    search_query: str
    filter: Union[str, None]

class ChatState(TypedDict):
    task_id: str
    user_input: str
    current_results: List[SearchResult]
    vetted_results: List[SearchResult]
    discarded_results: List[SearchResult]
    processed_ids: Set[str]
    reviews: List[str]
    decisions: List[str]
    final_answer: Union[str, None]
    attempts: int
    search_history: List[Dict[str, Any]]
    thought_process: List[Dict[str, Any]]


# --- LLM & Embeddings Setup ---
llm = AzureChatOpenAI(
    azure_deployment=aoai_deployment,
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint
)
review_llm = llm.with_structured_output(ReviewDecision)

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint
)

# --- Helper Functions ---
def format_search_results(results: List[SearchResult]) -> str:
    output_parts = ["\n=== Search Results ==="]
    for i, result in enumerate(results, 0):
        result_parts = [
            f"\nResult #{i}",
            "=" * 80,
            f"ID: {result['id']}",
            f"Source File: {result['source_file']}",
            f"Source Pages: {result['source_pages']}",
            "\n<Start Content>",
            "-" * 80,
            result['content'],
            "-" * 80,
            "<End Content>"
        ]
        output_parts.extend(result_parts)
    return "\n".join(output_parts)

def run_search(search_query: str, processed_ids: Set[str], category_filter: Union[str, None] = None) -> List[SearchResult]:
    query_vector = embeddings_model.embed_query(search_query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=K_NEAREST_NEIGHBORS,
        fields="content_vector"
    )
    filter_parts = []
    if processed_ids:
        ids_string = ','.join(processed_ids)
        filter_parts.append(f"not search.in(id, '{ids_string}')")
    if category_filter:
        filter_parts.append(f"({category_filter})")
    filter_str = " and ".join(filter_parts) if filter_parts else None
    results = search_client.search(
        search_text=search_query,
        vector_queries=[vector_query],
        filter=filter_str,
        select=["id", "content", "source_file", "source_pages"],
        top=NUM_SEARCH_RESULTS
    )
    search_results = []
    for result in results:
        search_result = SearchResult(
            id=result["id"],
            content=result["content"],
            source_file=result["source_file"],
            source_pages=result["source_pages"],
            score=result["@search.score"]
        )
        search_results.append(search_result)
    return search_results

# --- Pipeline Functions ---
def generate_search_query(state: ChatState) -> ChatState:
    state["attempts"] += 1
    llm_input = f"User Question: {state['user_input']}"
    messages = [
        {"role": "system", "content": query_prompt},
        {"role": "user", "content": llm_input}
    ]
    llm_with_search_prompt = llm.with_structured_output(SearchPromptResponse)
    search_response = llm_with_search_prompt.invoke(messages)
    state["search_history"].append({
        "query": search_response.search_query,
        "filter": search_response.filter
    })
    current_results = run_search(
        search_query=search_response.search_query,
        processed_ids=state["processed_ids"],
        category_filter=search_response.filter
    )
    state["current_results"] = current_results
    state["thought_process"].append({
        "type": "retrieve",
        "details": {
            "user_question": state["user_input"],
            "generated_search_query": search_response.search_query,
            "filter": search_response.filter,
            "results_summary": [
                {
                    "source_file": res["source_file"],
                    "source_pages": res["source_pages"]
                } for res in current_results
            ]
        }
    })
    service_bus_handler.publish_event_sync(
        state["task_id"],
        "retrieve",
        {
            "message": "Searching for documents...",
            "search_query": search_response.search_query,
            "filter": search_response.filter,
            "results_count": len(current_results)
        }
    )
    return state

def review_results(state: ChatState) -> ChatState:
    current_results_formatted = format_search_results(state["current_results"]) if state["current_results"] else "No current results."
    vetted_results_formatted = format_search_results(state["vetted_results"]) if state["vetted_results"] else "No previously vetted results."
    llm_input = f"""
User Question: {state['user_input']}

<Current Search Results to review>
{current_results_formatted}
<end current search results to review>

<Previously vetted results (do not review)>
{vetted_results_formatted}
<end previously vetted results>
"""
    review_prompt = (
        "Review these search results and determine which contain relevant information to answer the user's question.\n\n"
        "Respond with:\n"
        "1. thought_process: Your analysis of the results.\n"
        "2. valid_results: List of indices (0-N) for useful results.\n"
        "3. invalid_results: List of indices (0-N) for irrelevant results.\n"
        "4. decision: Either \"retry\" or \"finalize\"."
    )
    messages = [
        {"role": "system", "content": review_prompt},
        {"role": "user", "content": llm_input}
    ]
    review = review_llm.invoke(messages)
    state["thought_process"].append({
        "type": "review",
        "details": {
            "review_thought_process": review.thought_process,
            "decision": review.decision
        }
    })
    state["reviews"].append(review.thought_process)
    state["decisions"].append(review.decision)
    for idx in review.valid_results:
        result = state["current_results"][idx]
        state["vetted_results"].append(result)
        state["processed_ids"].add(result["id"])
    for idx in review.invalid_results:
        result = state["current_results"][idx]
        state["discarded_results"].append(result)
        state["processed_ids"].add(result["id"])
    state["current_results"] = []
    service_bus_handler.publish_event_sync(
        state["task_id"],
        "review",
        {
            "message": "Reviewing the documents...",
            "decision": review.decision,
            "review_thought_process": review.thought_process
        }
    )
    return state

def review_router(state: ChatState) -> str:
    if state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    latest_decision = state["decisions"][-1]
    return "finalize" if latest_decision == "finalize" else "retry"

def finalize(state: ChatState) -> ChatState:
    final_prompt = "Create a comprehensive answer to the user's question using the vetted results. Make sure to respond in valid markdown. "
    llm_input = f"""

User Question: {state['user_input']}

Vetted Results:
{chr(10).join([f"- {r['content']}" for r in state["vetted_results"]])}

Synthesize these results into a clear, complete answer. If there were no vetted results, say you couldn't find any relevant information. Make sure to respond in valid markdown. Respond with the answer and then cite the sources. """
    

    messages = [
        {"role": "system", "content": final_prompt},
        {"role": "user", "content": llm_input}
    ]
    final_response = ""
    for chunk in llm.stream(messages):
        chunk_content = chunk.content
        print(f"Raw chunk: '{chunk_content}' (len={len(chunk_content)})")  # Debug logging
        final_response += chunk.content
        if chunk.content:
            service_bus_handler.publish_event_sync(
                state["task_id"], 
                "chunk_stream", 
                {"message": chunk.content}
            )
            #time.sleep(0.1)
    state["final_answer"] = final_response
    state["thought_process"].append({
        "type": "response",
        "details": {"final_answer": final_response}
    })
    final_payload = {
        "final_answer": final_response,
        "citations": state["vetted_results"],
        "thought_process": state["thought_process"]
    }
    service_bus_handler.publish_event_sync(
        state["task_id"],
        "final_payload",
        final_payload
    )
    return state

# --- Workflow Graph ---
from langgraph.graph import StateGraph, START, END

def build_graph() -> StateGraph:
    builder = StateGraph(ChatState)
    builder.add_node("generate_search_query", generate_search_query)
    builder.add_node("review_results", review_results)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "generate_search_query")
    builder.add_edge("generate_search_query", "review_results")
    builder.add_conditional_edges(
        "review_results",
        review_router,
        {"retry": "generate_search_query", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile()

# --- FastAPI & Service Bus Setup ---
connection_string = os.environ.get("AZURE_SERVICE_BUS_CONNECTION_STRING", "")
topic_name = os.environ.get("AZURE_SERVICE_BUS_TOPIC_NAME")
agent_id = "1"
agent_name = "data-intelligence-agent"
service_bus_handler = ServiceBusHandler(connection_string, topic_name, agent_id, agent_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service_bus_handler.initialize()
    yield
    await service_bus_handler.close()

app = FastAPI(
    title="Data Intelligence Agent",
    description="Data Intelligence Agent is a multi-step agent that uses agentic RAG to answer a given question.",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class RunAgentRequest(BaseModel):
    taskId: str
    message: str

async def process_task(task_id: str, user_message: str):
    if not service_bus_handler._is_initialized:
        await service_bus_handler.initialize()

    initial_state: ChatState = {
        "task_id": task_id,
        "user_input": user_message,
        "current_results": [],
        "vetted_results": [],
        "discarded_results": [],
        "processed_ids": set(),
        "reviews": [],
        "decisions": [],
        "final_answer": None,
        "attempts": 0,
        "search_history": [],
        "thought_process": []
    }
    try:
        graph = build_graph()
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
        if not final_state["final_answer"]:
            service_bus_handler.publish_event_sync(
                task_id,
                "final_payload",
                {"error": "Unable to find a satisfactory answer after maximum attempts."}
            )
    finally:
        # Wait for queued messages to be processed
        await service_bus_handler._message_queue.join()


@app.get("/health")
async def root():
    return {"status": "healthy"}

@app.post("/run_agent", status_code=202)
async def run_agent(request: RunAgentRequest):
    try:
        task_id = request.taskId
        user_message = request.message
        asyncio.create_task(process_task(task_id, user_message))
        return {"status": "accepted", "taskId": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing only.
async def main():
    """Local testing function only - not used in container app"""
    try:
        test_task_id = "621100"
        test_message = "What year was my house built?"
        await process_task(test_task_id, test_message)
        # Sleep here since we're doing local testing and need to wait for messages
        await asyncio.sleep(5)  
    finally:
        # Cleanup only needed for local testing
        await service_bus_handler.close()

if __name__ == "__main__":
    asyncio.run(main())
