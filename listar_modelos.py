from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent / ".env")
client = genai.Client()

print("Modelos disponíveis para gerar texto:\n")
for modelo in client.models.list():
    if "generateContent" in (modelo.supported_actions or []):
        print(f"- {modelo.name.removeprefix('models/')}  ({modelo.display_name})")
