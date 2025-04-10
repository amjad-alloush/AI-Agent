import os
from venv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Access environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")