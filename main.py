# main.py
import os
import json
import asyncio
from datetime import datetime, UTC

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Agent Run API",
    description="API for running asynchronous agent tasks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Azure Service Bus configuration
connection_string = os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
queue_name = os.getenv("AZURE_SERVICE_BUS_QUEUE_NAME")

class RunAgentRequest(BaseModel):
    taskId: str
    # Optionally, add additional fields for input data if needed
    # input_data: dict = {}

async def send_to_service_bus(message_data: dict):
    """Send a message to Azure Service Bus using the provided schema."""
    async with ServiceBusClient.from_connection_string(
        conn_str=connection_string,
        logging_enable=True
    ) as client:
        async with client.get_queue_sender(queue_name) as sender:
            message = ServiceBusMessage(json.dumps(message_data))
            await sender.send_messages(message)

async def process_task(task_id: str):
    """
    Simulate multi-step processing for a task.
    Each step publishes a partial event (and a final event) to the Service Bus.
    """
    # Step 1: Processing started
    event1 = {
        "taskId": task_id,
        "agentId": "chat-agent",
        "eventIndex": 1,
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "isFinal": False,
        "status": "in-progress",
        "eventType": "partial",
        "payload": {"message": "Processing started"}
    }
    await send_to_service_bus(event1)
    await asyncio.sleep(1)  # Simulate processing delay

    # Step 2: Intermediate step
    event2 = {
        "taskId": task_id,
        "agentId": "chat-agent",
        "eventIndex": 2,
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "isFinal": False,
        "status": "in-progress",
        "eventType": "partial",
        "payload": {"message": "Step 2 completed"}
    }
    await send_to_service_bus(event2)
    await asyncio.sleep(1)  # Simulate processing delay

    # Final Step: Task completed
    final_event = {
        "taskId": task_id,
        "agentId": "chat-agent",
        "eventIndex": 3,
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "isFinal": True,
        "status": "completed",
        "eventType": "final",
        "payload": {"message": "Task completed successfully"}
    }
    await send_to_service_bus(final_event)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/run_agent", status_code=202)
async def run_agent(request: RunAgentRequest):
    """
    Endpoint to initiate an agent task.
    Expects the frontend to provide a unique taskId.
    Immediately returns a 202 Accepted with the taskId,
    then processes the task asynchronously.
    """
    try:
        task_id = request.taskId
        
        # Kick off background processing for this task
        asyncio.create_task(process_task(task_id))
        
        # Immediately return the accepted response with the taskId
        return {
            "status": "accepted",
            "taskId": task_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def main():
    """Test function to simulate a complete agent run locally"""
    # Create a test request
    test_task_id = "test-task-123"
    request = RunAgentRequest(taskId=test_task_id)
    
    print(f"Starting test run with task ID: {test_task_id}")
    
    # Simulate the API call
    response = await run_agent(request)
    print(f"Initial response: {response}")
    
    # Wait for the background task to complete (5 seconds should be enough given the delays)
    await asyncio.sleep(5)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(main())