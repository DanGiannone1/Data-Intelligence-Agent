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

# Import OpenTelemetry for creating custom spans
from opentelemetry import trace
from opentelemetry.trace import get_tracer

print("App Insights Connection String:", os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))
configure_azure_monitor(connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"))

os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'] = 'true'
print("Content recording enabled:", os.environ['AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED'])

AIInferenceInstrumentor().instrument()

endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT", "https://djgaihub9630729362.services.ai.azure.com/models")
model_name = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
FOUNDRY_API_KEY = os.getenv("FOUNDRY_API_KEY", "MISSING_KEY")

print("Endpoint:", endpoint)
print("FOUNDRY_API_KEY:", FOUNDRY_API_KEY)

tracer = get_tracer(__name__)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(FOUNDRY_API_KEY)
)

# ---------------------------------------------------------------------
# Custom Tool Function (Separate from LLMs)
# ---------------------------------------------------------------------
def my_custom_tool(city: str) -> str:
    """Custom tool function that processes city data."""
    return f"The custom tool processed city: {city}"

def trace_tool_call(tracer, function_name, arguments, result):
    """Helper function to create a span for a tool call."""
    with tracer.start_as_current_span(f"ToolCall: {function_name}") as span:
        # Add attributes to identify it as a tool call
        span.set_attribute("llm.tool_call.name", function_name)
        span.set_attribute("llm.tool_call.arguments", str(arguments))
        span.set_attribute("llm.tool_call.result", str(result))
        span.set_attribute("span_type", "ToolCall")  # Explicitly mark it as a ToolCall
        print(f"Traced tool call: {function_name}, Arguments: {arguments}, Result: {result}")
        return result

# ---------------------------------------------------------------------
# Existing Functions for Weather and Temperature
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def main():
    tracer = trace.get_tracer(__name__)

    # Wrap the entire script in one top-level span, so everything is one trace
    with tracer.start_as_current_span("LLM Conversation Trace"):

        # ---------------------------------------------------------------------
        # FIRST LLM CALL
        # ---------------------------------------------------------------------
        with tracer.start_as_current_span("LLM: chat gpt-4 (Seattle weather)"):
            weather_tool = ChatCompletionsToolDefinition(
                function=FunctionDefinition(
                    name="get_weather",
                    description="Returns weather for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name for weather lookup.",
                            }
                        },
                        "required": ["city"],
                    },
                )
            )

            temperature_tool = ChatCompletionsToolDefinition(
                function=FunctionDefinition(
                    name="get_temperature",
                    description="Returns temperature for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name for temperature lookup.",
                            }
                        },
                        "required": ["city"],
                    },
                )
            )

            messages = [
                SystemMessage("You are a helpful AI assistant."),
                UserMessage("What is the weather and temperature in Seattle right now?")
            ]

            response = client.complete(
                messages=messages,
                tools=[weather_tool, temperature_tool],
                model=model_name
            )

        # ---------------------------------------------------------------------
        # CUSTOM FUNCTION-CALL SPAN (In between LLM calls)
        # ---------------------------------------------------------------------
        with tracer.start_as_current_span("Function: get_weather_for_seattle"):
            weather = get_weather("Seattle")
            print("Manual function call result:", weather)

        # ---------------------------------------------------------------------
        # CUSTOM TOOL CALL (Standalone)
        # ---------------------------------------------------------------------
        with tracer.start_as_current_span("Standalone Tool Call Example"):
            city = "Seattle"
            result = my_custom_tool(city)

            # Manually trace this as a tool call
            trace_tool_call(
                tracer=tracer,
                function_name="my_custom_tool",
                arguments={"city": city},
                result=result
            )

        # ---------------------------------------------------------------------
        # SECOND LLM CALL
        # ---------------------------------------------------------------------
        with tracer.start_as_current_span("LLM: chat gpt-4 (Thailand travel)"):
            messages = [
                SystemMessage("You are a helpful AI assistant."),
                UserMessage("What are some places to visit in Thailand?")
            ]
            response = client.complete(
                messages=messages,
                model=model_name
            )

            print("Assistant says:", response.choices[0].message.content)

if __name__ == "__main__":
    main()
