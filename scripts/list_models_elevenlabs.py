import os
import sys
import httpx
from pipeline.config import PipelineConfig

def main():
    config = PipelineConfig("config.yaml")
    url = "https://api.elevenlabs.io/v1/models"
    headers = {"xi-api-key": config.elevenlabs_api_key}
    
    try:
        with httpx.Client(verify=False) as client:
            response = client.get(url, headers=headers)
            if response.status_code == 200:
                models = response.json()
                print("\nAvailable ElevenLabs Models for your key:")
                for m in models:
                    print(f"- Name: {m.get('name')} | ID: {m.get('model_id')}")
            else:
                print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Failed to query models: {e}")

if __name__ == "__main__":
    main()
