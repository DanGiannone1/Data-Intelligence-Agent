# service_bus_handler.py

import os
import json
import asyncio
import logging
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("azure").setLevel(logging.WARNING)

class ServiceBusHandler:
    def __init__(self, connection_string: str, topic_name: str):
        self.connection_string = connection_string.strip()
        if self.connection_string.startswith("\ufeff"):
            self.connection_string = self.connection_string.replace("\ufeff", "")
        self.topic_name = topic_name
        self.sb_client = None
        self.sender = None
        self.lock = asyncio.Lock()  # To avoid race conditions on reinitialization

    async def initialize(self):
        """Initialize the Service Bus client and sender."""
        logger.info("Starting Service Bus connection...")
        self.sb_client = ServiceBusClient.from_connection_string(
            conn_str=self.connection_string, logging_enable=True
        )
        self.sender = self.sb_client.get_topic_sender(topic_name=self.topic_name)
        await self.sender.__aenter__()
        logger.info("Service Bus connection established.")

    async def close(self):
        """Gracefully close the Service Bus sender and client."""
        if self.sender:
            await self.sender.__aexit__(None, None, None)
        if self.sb_client:
            await self.sb_client.__aexit__(None, None, None)
        logger.info("Service Bus connection closed.")

    async def get_sender(self):
        """
        Ensure that the sender is available. If not, reinitialize.
        Uses a lock to avoid simultaneous reinitialization.
        """
        async with self.lock:
            if self.sender is None:
                logger.info("Reinitializing Service Bus sender...")
                await self.initialize()
        return self.sender

    async def send_message(self, message_data: dict, retries: int = 1):
        """
        Send a message to Azure Service Bus.
        If an error occurs, try to reinitialize the sender and retry.
        """
        event_index = message_data.get("eventIndex")
        task_id = message_data.get("taskId")
        # Log each file/message sent to Service Bus
        logger.info(f"Sending message for task {task_id}, event {event_index}...")
        
        msg = ServiceBusMessage(
            json.dumps(message_data),
            session_id=task_id
        )
        try:
            sender = await self.get_sender()
            await sender.send_messages(msg)
            logger.info(f"Message for task {task_id}, event {event_index} sent successfully.")
        except Exception as e:
            logger.error(f"Error sending message for task {task_id}, event {event_index}: {e}")
            # If there is an error, tear down the sender and retry if possible.
            async with self.lock:
                if self.sender:
                    try:
                        await self.sender.__aexit__(None, None, None)
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup error: {cleanup_error}")
                    self.sender = None
            if retries > 0:
                await self.send_message(message_data, retries=retries - 1)
            else:
                raise
