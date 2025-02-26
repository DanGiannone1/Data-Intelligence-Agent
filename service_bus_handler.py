import os
import json
import asyncio
from datetime import datetime, timezone
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

class ServiceBusHandler:
    def __init__(self, connection_string: str, topic_name: str, agent_id: str, agent_name: str):
        self.connection_string = connection_string.strip().replace("\ufeff", "")
        self.topic_name = topic_name
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.sb_client = None
        self.sender = None
        self.lock = asyncio.Lock()
        self._message_queue = asyncio.Queue()
        self._dispatcher_task = None
        self.event_counter = 0
        self._chunk_buffers = {}
        self.update_loop = None
        self._is_initialized = False

    def _get_next_event_index(self) -> int:
        self.event_counter += 1
        return self.event_counter

    async def initialize(self):
        if self._is_initialized:
            return
            
        # Store the loop
        self.update_loop = asyncio.get_event_loop()
            
        print("Starting Service Bus connection...")
        self.sb_client = ServiceBusClient.from_connection_string(
            conn_str=self.connection_string, logging_enable=False
        )
        self.sender = self.sb_client.get_topic_sender(topic_name=self.topic_name)
        await self.sender.__aenter__()
        print("Service Bus connection established.")

        # Start the dispatcher task
        if not self._dispatcher_task or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(self._dispatch_messages())
            print("Dispatcher task started")
        
        self._is_initialized = True

    def publish_event_sync(self, task_id: str, event_type: str, payload: dict):
        """Thread-safe synchronous method to publish events"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If we're in a different thread, we need to use the stored loop
            if not self.update_loop:
                raise Exception("ServiceBusHandler not initialized!")
            loop = self.update_loop

        # Create the event data
        event = {
            "taskId": task_id,
            "agentId": self.agent_id,
            "agentName": self.agent_name,
            "eventType": event_type,
            "payload": payload
        }

        # Put directly into queue
        loop.call_soon_threadsafe(self._message_queue.put_nowait, event)
        print(f"[ENQUEUE] Task {task_id} | {event_type} | {json.dumps(payload)}")

    async def enqueue_event(self, task_id: str, event_type: str, payload: dict):
        if not self._is_initialized:
            await self.initialize()
            
        event = {
            "taskId": task_id,
            "agentId": self.agent_id,
            "agentName": self.agent_name,
            "eventType": event_type,
            "payload": payload
        }
        await self._message_queue.put(event)
        print(f"[ENQUEUE] Task {task_id} | {event_type} | {json.dumps(payload)}")

    async def close(self):
        # Wait for message queue to be empty before closing
        if not self._message_queue.empty():
            print("Waiting for message queue to be processed...")
            await self._message_queue.join()
            await asyncio.sleep(1)  # Give a small buffer after queue is empty
        
        # Flush any pending chunk buffers before closing
        for task_id, buffered_text in self._chunk_buffers.items():
            if buffered_text:
                batched_event = {
                    "taskId": task_id,
                    "agentId": self.agent_id,
                    "agentName": self.agent_name,
                    "eventType": "chunk_stream",
                    "payload": {"message": buffered_text}
                }
                print(f"[BUFFER] Task {task_id} | Final flush: '{buffered_text}'")
                await self._send_with_retries(batched_event)
        self._chunk_buffers.clear()

        if self._dispatcher_task:
            print("Cancelling dispatcher task...")
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                print("Dispatcher task cancelled successfully")

        if self.sender:
            print("Closing Service Bus sender...")
            await self.sender.__aexit__(None, None, None)
        if self.sb_client:
            print("Closing Service Bus client...")
            await self.sb_client.__aexit__(None, None, None)
        
        self._is_initialized = False
        print("Service Bus handler closed successfully")

    async def get_sender(self):
        async with self.lock:
            if not self.sender:
                print("Reinitializing Service Bus sender...")
                await self.initialize()
        return self.sender

    async def _dispatch_messages(self):
        print("Message dispatcher started and waiting for messages...")
        while True:
            try:
                message_data = await self._message_queue.get()
                event_type = message_data.get("eventType")
                task_id = message_data.get("taskId")
                
                try:
                    if event_type == "chunk_stream":
                        # Always add chunk to buffer
                        chunk_text = message_data["payload"].get("message", "")
                        print(f"[BUFFER DEBUG] Task {task_id} | Raw chunk: '{chunk_text}' (len={len(chunk_text)})")
                        if task_id in self._chunk_buffers:
                            current_buffer = self._chunk_buffers[task_id]
                            print(f"[BUFFER DEBUG] Task {task_id} | Current buffer before: '{current_buffer}' (len={len(current_buffer)})")
                            self._chunk_buffers[task_id] += chunk_text
                            print(f"[BUFFER DEBUG] Task {task_id} | Buffer after: '{self._chunk_buffers[task_id]}' (len={len(self._chunk_buffers[task_id])})")
                        else:
                            self._chunk_buffers[task_id] = chunk_text
                        
                        buffer_content = self._chunk_buffers[task_id]
                        print(f"[BUFFER] Task {task_id} | Added '{chunk_text}' | Current: '{buffer_content}'")
                        
                        # Flush if buffer is large enough
                        if len(buffer_content) >= 8:
                            buffered_text = self._chunk_buffers.pop(task_id)
                            batched_event = {
                                "taskId": task_id,
                                "agentId": self.agent_id,
                                "agentName": self.agent_name,
                                "eventType": "chunk_stream",
                                "payload": {"message": buffered_text}
                            }
                            print(f"[BUFFER] Task {task_id} | Flushing: '{buffered_text}'")
                            await self._send_with_retries(batched_event)
                    else:
                        # For non-chunk events, flush any existing buffer first
                        if task_id in self._chunk_buffers and self._chunk_buffers[task_id]:
                            buffered_text = self._chunk_buffers.pop(task_id)
                            batched_event = {
                                "taskId": task_id,
                                "agentId": self.agent_id,
                                "agentName": self.agent_name,
                                "eventType": "chunk_stream",
                                "payload": {"message": buffered_text}
                            }
                            print(f"[BUFFER] Task {task_id} | Flushing: '{buffered_text}'")
                            await self._send_with_retries(batched_event)
                        
                        # Then send the non-chunk event
                        await self._send_with_retries(message_data)
                
                except Exception as e:
                    print(f"Error processing message for task {task_id}: {str(e)}")
                
                finally:
                    self._message_queue.task_done()
            
            except asyncio.CancelledError:
                print("Dispatcher received cancellation signal")
                break
            except Exception as e:
                print(f"Unexpected error in dispatcher: {str(e)}")
                continue

    async def _send_with_retries(self, message_data: dict, retries: int = 3):
        # Add eventIndex and timestamp right before sending
        message_data = {
            **message_data,
            "eventIndex": self._get_next_event_index(),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
        
        task_id = message_data.get("taskId")
        event_type = message_data.get("eventType")
        event_index = message_data.get("eventIndex")
        message = message_data.get("payload", {}).get("message", "")
        print(f"[SEND] Task {task_id} | Event {event_index} | {event_type} | {message}")
        
        msg = ServiceBusMessage(
            json.dumps(message_data),
            session_id=task_id
        )

        for attempt in range(retries):
            try:
                sender = await self.get_sender()
                await sender.send_messages(msg)
                print(f"[SUCCESS] {json.dumps(message_data)}")
                return
            except Exception as e:
                print(f"Error sending message for task {task_id}, event {event_index} (attempt {attempt + 1}/{retries}): {str(e)}")
                
                if attempt < retries - 1:  # Don't wait after the last attempt
                    async with self.lock:
                        if self.sender:
                            try:
                                await self.sender.__aexit__(None, None, None)
                            except Exception as cleanup_error:
                                print(f"Error during sender cleanup: {str(cleanup_error)}")
                            self.sender = None
                    await asyncio.sleep(0.5 * (attempt + 1))  # Progressive backoff
                else:
                    raise Exception(f"Failed to send message for task {task_id}, event {event_index} after {retries} attempts")