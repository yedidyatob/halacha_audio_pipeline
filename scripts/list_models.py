import os
import sys
from google import genai
from pipeline.config import PipelineConfig

def main():
    config = PipelineConfig("config.yaml")
    client = genai.Client(api_key=config.gemini_api_key)
    print("Listing available models for the configured key:")
    try:
        # Try to list models
        models = client.models.list()
        for m in models:
            print(f"- {m.name} (Supported: {m.supported_actions})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    main()
