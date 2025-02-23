import os
import json
import asyncio
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

class ServiceBusHandler:
    def __init__(self, connection_string: str, topic_name: str):
        self.connection_string = connection_string.strip()
        if self.connection_string.startswith("\ufeff"):
            self.connection_string = self.connection_string.replace("\ufeff", "")
        self.topic_name = topic_name
        self.sb_client = None
        self.sender = None
        self.lock = asyncio.Lock()  # To avoid race conditions on reinitialization
        self._message_queue = asyncio.Queue()
        self._dispatcher_task = None

    async def initialize(self):
        print("Starting Service Bus connection...")
        self.sb_client = ServiceBusClient.from_connection_string(
            conn_str=self.connection_string, logging_enable=False
        )
        self.sender = self.sb_client.get_topic_sender(topic_name=self.topic_name)
        await self.sender.__aenter__()
        print("Service Bus connection established.")

        # Start the dispatcher task if it hasn't been started yet.
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_messages())

    async def close(self):
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        if self.sender:
            await self.sender.__aexit__(None, None, None)
        if self.sb_client:
            await self.sb_client.__aexit__(None, None, None)
        print("Service Bus connection closed.")

    async def get_sender(self):
        async with self.lock:
            if self.sender is None:
                print("Reinitializing Service Bus sender...")
                await self.initialize()
        return self.sender

    async def enqueue_message(self, message_data: dict):
        # Enqueue the message to be sent by the dispatcher.
        await self._message_queue.put(message_data)
        print(f"Enqueued message for task {message_data.get('taskId')}, event {message_data.get('eventIndex')}.")

    async def _dispatch_messages(self):
        batch_buffer = None  # Used to accumulate chunk_stream messages.
        while True:
            message_data = await self._message_queue.get()
            try:
                if message_data.get("eventType") == "chunk_stream":
                    # Accumulate the chunk_stream event.
                    if batch_buffer is None:
                        # Start a new batch.
                        batch_buffer = message_data.copy()
                        # Make a deep copy of the payload.
                        batch_buffer["payload"] = message_data["payload"].copy()
                    else:
                        # Append the new chunk.
                        batch_buffer["payload"]["message"] += message_data["payload"]["message"]
                    # Mark the chunk as processed.
                    self._message_queue.task_done()
                    # If the batch has reached the minimum length, send it.
                    if len(batch_buffer["payload"]["message"]) >= 8:
                        await self._send_with_retries(batch_buffer)
                        batch_buffer = None
                else:
                    # For a non-chunk_stream event, flush any pending chunk batch.
                    if batch_buffer is not None:
                        await self._send_with_retries(batch_buffer)
                        batch_buffer = None
                    await self._send_with_retries(message_data)
                    self._message_queue.task_done()
            except Exception as e:
                print(f"Failed to send message: {e}")
                self._message_queue.task_done()

    async def _send_with_retries(self, message_data: dict, retries: int = 3):
        event_index = message_data.get("eventIndex")
        print(f"Sending message for task {message_data.get('taskId')}, event {event_index}...")
        msg = ServiceBusMessage(
            json.dumps(message_data),
            session_id=message_data.get("taskId")
        )
        for attempt in range(retries):
            try:
                sender = await self.get_sender()
                await sender.send_messages(msg)
                print(f"Message for task {message_data.get('taskId')}, event {event_index} sent successfully.")
                return
            except Exception as e:
                print(f"Error sending message for task {message_data.get('taskId')}, event {event_index} (attempt {attempt+1}): {e}")
                async with self.lock:
                    if self.sender:
                        try:
                            await self.sender.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            print(f"Cleanup error: {cleanup_error}")
                        self.sender = None
                await asyncio.sleep(0.5)
        raise Exception(f"Failed to send message for task {message_data.get('taskId')}, event {event_index} after {retries} attempts.")  