import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Azure SDK imports
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage
)
from azure.core.credentials import AzureKeyCredential

# OpenTelemetry imports
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.ai.inference.tracing import AIInferenceInstrumentor
from opentelemetry import trace

# Import the Service Bus handler
from service_bus_handler import ServiceBusHandler

# Load environment variables
load_dotenv()

# Configure Azure OpenAI
inference_endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
model_name = os.getenv("DEPLOYMENT_NAME")
api_key = os.getenv("FOUNDRY_API_KEY")

# Configure tracing
configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)
AIInferenceInstrumentor().instrument()

# Initialize chat client
chat_client = ChatCompletionsClient(
    endpoint=inference_endpoint,
    credential=AzureKeyCredential(api_key)
)

# --- Type Definitions ---
class RunAgentRequest(BaseModel):
    taskId: str
    message: str

def get_llm_response(user_message: str) -> str:
    """Single LLM call to process the input"""
    messages = [
        SystemMessage("You are a helpful assistant."),
        UserMessage(user_message)
    ]
    
    response = chat_client.complete(
        messages=messages,
        model=model_name
    )
    
    return response.choices[0].message.content

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
    title="Data Intelligence Agent API",
    description="Simple agent with LLM integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

async def process_task(task_id: str, user_message: str):
    if not service_bus_handler._is_initialized:
        await service_bus_handler.initialize()

    try:
        # Starting event
        service_bus_handler.publish_event_sync(
            task_id,
            "start",
            {"message": "Starting task..."}
        )
        
        # Process with LLM
        service_bus_handler.publish_event_sync(
            task_id,
            "process",
            {"message": "Calling LLM..."}
        )
        
        response = await asyncio.to_thread(get_llm_response, user_message)
        
        # Final payload
        service_bus_handler.publish_event_sync(
            task_id,
            "final_payload",
            {"response": response}
        )
            
    except Exception as e:
        service_bus_handler.publish_event_sync(
            task_id,
            "error",
            {"error": str(e)}
        )
        raise
    finally:
        await service_bus_handler._message_queue.join()

@app.get("/")
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

# For local testing only
async def main():
    """Local testing function - not used in container app"""
    try:
        test_task_id = "test_123"
        test_message = "Write me a haiku about cats"
        await process_task(test_task_id, test_message)
        # Sleep briefly since we're doing local testing
        await asyncio.sleep(5)
    finally:
        # Cleanup needed for local testing
        await service_bus_handler.close()

if __name__ == "__main__":
    asyncio.run(main())