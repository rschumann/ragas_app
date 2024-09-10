import os
import logging
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()
    # Verify that critical environment variables are set
    required_vars = ['OPENAI_API_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_HOST', 'API_URL', 'PROJECT']
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)
