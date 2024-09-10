import os
import logging
from dotenv import load_dotenv

def setup_environment():
    load_dotenv()
    required_vars = ['OPENAI_API_KEY', 'LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_HOST', 'API_URL', 'PROJECT']
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise EnvironmentError(f"Missing required environment variable: {var}")
        print(f"{var}: {'*' * len(value)}")  # Print asterisks instead of actual value for security

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
