import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini

# Load environment variables from the parent directory's .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Configure the LLM for the inference task
# A powerful model is needed for nuanced semantic understanding.
LLM = Gemini(
    id=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
    api_key=os.getenv("GEMINI_API_KEY")
)

# Define the specialized Inference Agent
inference_agent = Agent(
    name="InferenceAgent",
    role="Semantic Relevance Classifier",
    model=LLM,
    instructions=[
        "Your sole purpose is to determine if a given 'word' is semantically related to a given 'topic'.",
        "You must respond ONLY with a single, raw, valid JSON object.",
        "The JSON object must have a single key, 'is_related', with a boolean value (true or false).",
        "Do not provide any explanation or extra text. Just the JSON.",
        "Example:",
        "Input: topic='치과', word='임플란트'",
        "Output: {\"is_related\": true}",
        "Input: topic='치과', word='컴퓨터'",
        "Output: {\"is_related\": false}",
    ],
    # We don't need tool calls or markdown for this simple, focused agent.
    show_tool_calls=False,
    markdown=False,
) 