# main.py

import os
import json
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service_bus_handler import ServiceBusHandler

# Setup basic logging configuration for the application.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("azure").setLevel(logging.WARNING)

# Initialize the ServiceBusHandler instance with config.
connection_string = os.environ.get("AZURE_SERVICE_BUS_CONNECTION_STRING", "")
topic_name = os.environ.get("AZURE_SERVICE_BUS_TOPIC_NAME")
service_bus_handler = ServiceBusHandler(connection_string, topic_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await service_bus_handler.initialize()
    yield
    # Shutdown
    await service_bus_handler.close()

app = FastAPI(
    title="Agent Run API",
    description="API for running asynchronous agent tasks",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (adjust origins as needed)
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
    """
    Simulate multi-step processing for a task by sending several events to Service Bus.
    """
    now = lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events = [
        {
            "taskId": task_id,
            "agentId": "1",
            "agentName": "data-intelligence-agent",
            "eventIndex": 1,
            "timestamp": now(),
            "eventType": "retrieve",
            "payload": {"message": "Searching for documents..."}
        },
        {
            "taskId": task_id,
            "agentId": "1",
            "agentName": "data-intelligence-agent",
            "eventIndex": 2,
            "timestamp": now(),
            "eventType": "review",
            "payload": {"message": "Reviewing documents..."}
        },
        {
            "taskId": task_id,
            "agentId": "1",
            "agentName": "data-intelligence-agent",
            "eventIndex": 3,
            "timestamp": now(),
            "eventType": "chunk_stream",
            "payload": {"message": "abc"}
        },
        {
            "taskId": task_id,
            "agentId": "1",
            "agentName": "data-intelligence-agent",
            "eventIndex": 4,
            "timestamp": now(),
            "eventType": "final_payload",
            "payload": {"message": f"Task completed successfully for: {user_message}"}
        }
    ]
    
    for event in events:
        logger.info(f"Processing task {task_id}: event {event.get('eventIndex')} - {event['payload']['message']}")
        await service_bus_handler.send_message(event)
    
    logger.info(f"Task {task_id} processing complete.")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/run_agent", status_code=202)
async def run_agent(request: RunAgentRequest):
    """
    Initiate an agent task.
    Expects a unique taskId and a message.
    Immediately returns a 202 Accepted with the taskId,
    then processes the task asynchronously.
    """
    try:
        task_id = request.taskId
        user_message = request.message
        logger.info(f"Received run_agent request for task {task_id} with message: {user_message}")
        # Launch background processing for this task.
        asyncio.create_task(process_task(task_id, user_message))
        return {"status": "accepted", "taskId": task_id}
    except Exception as e:
        logger.error(f"Error in run_agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# For local testing only.
async def main():
    test_task_id = "1234567890"
    test_message = "Hello, test!"
    logger.info(f"Starting test run with task ID: {test_task_id}")
    await process_task(test_task_id, test_message)
    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
