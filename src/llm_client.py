from google import genai
import os
from dotenv import load_dotenv

# Load API key (in .env file)
load_dotenv() 
api_key = os.getenv("API_KEY")
if api_key:
    API_KEY = api_key
else:
    print("API key not found in environmental variables")


MODEL_ID = "gemma-3-27b-it"
client = genai.Client(api_key=API_KEY)

def generate_translation(prompt):
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0),    # Temp to 0 to try and get the answer most probable 
        )

        return response.text.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return ""