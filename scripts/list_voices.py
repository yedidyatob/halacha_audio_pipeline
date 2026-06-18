import os
import sys
import httpx
from pipeline.config import PipelineConfig

def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config '{config_path}' not found.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    if not config.elevenlabs_api_key:
        print("Error: ElevenLabs API key not found in config.")
        sys.exit(1)
        
    print("Listing available ElevenLabs voices...")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": config.elevenlabs_api_key}
    
    try:
        # Using verify=False to bypass certificate errors
        with httpx.Client(verify=False) as client:
            response = client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                voices = data.get("voices", [])
                print(f"\nFound {len(voices)} voices:")
                for voice in voices:
                    print(f"- Name: {voice.get('name')} | ID: {voice.get('voice_id')} | Category: {voice.get('category')}")
            else:
                print(f"Error: Status code {response.status_code}, Body: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
