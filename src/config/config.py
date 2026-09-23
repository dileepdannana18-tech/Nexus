import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Config:
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    DEFAULT_URLS = [
    "https://example.com" # Or a very short webpage
]

    @staticmethod
    def get_llm():
        """Initialize and return the Gemini chat model"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key is not set in the environment variables.")
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )