from google import genai
import os
from dotenv import load_dotenv


# Accessing API KEY in environment variables
load_dotenv() 
api_key = os.getenv("API_KEY")
if api_key:
    API_KEY = api_key
print(api_key)

MODEL_ID = "gemma-3-27b-it"

client = genai.Client(api_key=API_KEY)
prompt = f"Write a very short but funny story of 4 or 5 sentences set in the XII century in France."

try:
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95
        ),
    )
    
    print(f"story: {response.text.strip()}")
    print(f"tokens: {response.usage_metadata.total_token_count}")

    print("-" * 50 + "\n")
    
except Exception as e:
    print(f"error in request: {e}")