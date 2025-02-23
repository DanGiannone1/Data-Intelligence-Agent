#foundry.py

import os
from dotenv import load_dotenv
load_dotenv()

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    CompletionsFinishReason,
    ToolMessage,
    AssistantMessage,
    ChatCompletionsToolCall,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
)
from azure.core.credentials import AzureKeyCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.ai.inference.tracing import AIInferenceInstrumentor

# Import OpenTelemetry for creating a parent span
from opentelemetry import trace

# 1) Print out the Application Insights connection string for debugging
print("App Insights Connection String:", os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))

# 2) Configure Application Insights
configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)

# 3) Enable detailed content recording for tracing (optional)
os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'] = 'true'
print("Content recording enabled:", os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'])

# 4) Instrument the AI Inference client (OpenTelemetry)
AIInferenceInstrumentor().instrument()

# 5) Set up client using environment-based values
endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
model_name = os.getenv("DEPLOYMENT_NAME")
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY")

print("Endpoint:", endpoint)
print("FOUNDRY_API_KEY:", FOUNDRY_API_KEY or "Not set")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(FOUNDRY_API_KEY or "MISSING_KEY")
)

# Optional: define a couple of Python "functions" to mimic the idea of function calls in the chat
def get_weather(city: str) -> str:
    """Dummy function that returns a made-up weather description for a city."""
    if city.lower() == "seattle":
        return "Seattle has a mild, rainy climate."
    elif city.lower() == "new york city":
        return "New York City has a humid subtropical climate."
    return f"Weather data not found for city: {city}"

def get_temperature(city: str) -> str:
    """Dummy function that returns a made-up temperature for a city."""
    if city.lower() == "seattle":
        return "65"
    elif city.lower() == "new york city":
        return "75"
    return "Unavailable"

def main():
    """
    Sample main function that:
      - Defines two 'tool' function definitions.
      - Sends prompts that might cause the model to call those functions.
      - Handles any function calls and finalizes answers.
      - Wraps all LLM calls in a single parent span to consolidate tracing.
    """

    # Get a tracer for the current module
    tracer = trace.get_tracer(__name__)

    # Create a parent span that will wrap the entire conversation
    with tracer.start_as_current_span("LLM Conversation Trace"):
        # Describe tools so the model is allowed to call them
        weather_tool = ChatCompletionsToolDefinition(
            function=FunctionDefinition(
                name="get_weather",
                description="Returns a brief weather description for a specified city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the weather info for.",
                        }
                    },
                    "required": ["city"],
                },
            )
        )

        temperature_tool = ChatCompletionsToolDefinition(
            function=FunctionDefinition(
                name="get_temperature",
                description="Returns the current temperature for the specified city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the temperature info for.",
                        }
                    },
                    "required": ["city"],
                },
            )
        )

        # First Chat conversation
        messages = [
            SystemMessage("You are a helpful AI assistant."),
            UserMessage("What is the weather and temperature in Seattle right now?")
        ]

        # Make the first chat completion call
        response = client.complete(
            messages=messages,
            tools=[weather_tool, temperature_tool],
            model=model_name
        )

        # If the model decides to call a function:
        if response.choices and response.choices[0].finish_reason == CompletionsFinishReason.TOOL_CALLS:
            # Append the model's function call message to the conversation
            messages.append(AssistantMessage(tool_calls=response.choices[0].message.tool_calls))

            # For each tool call, parse arguments and invoke the actual Python function
            if response.choices[0].message.tool_calls:
                import json
                for tool_call in response.choices[0].message.tool_calls:
                    if isinstance(tool_call, ChatCompletionsToolCall):
                        call_args = json.loads(tool_call.function.arguments.replace("'", '"'))
                        # Call the local Python function by name
                        fn = globals().get(tool_call.function.name)
                        if fn:
                            function_result = fn(**call_args)
                        else:
                            function_result = f"No local function found for {tool_call.function.name}"
                        print(f"Function call: {tool_call.function.name}({call_args}) -> {function_result}")

                        # Append the tool response so the model can consume it
                        messages.append(ToolMessage(function_result, tool_call_id=tool_call.id))

                # Now the model has additional data to finalize an answer
                response = client.complete(
                    messages=messages,
                    tools=[weather_tool, temperature_tool],
                    model=model_name
                )

        # Print final output for the first conversation
        if response.choices:
            print("Assistant says:", response.choices[0].message.content)
        else:
            print("No response from model")

        # Second Chat conversation (still within the same parent span)
        messages = [
            SystemMessage("You are a helpful AI assistant."),
            UserMessage("What are some places to visit in Thailand?")
        ]

        # Make the second chat completion call
        response = client.complete(
            messages=messages,
            model=model_name
        )

        print("Assistant says:", response.choices[0].message.content)


if __name__ == "__main__":
    main()
