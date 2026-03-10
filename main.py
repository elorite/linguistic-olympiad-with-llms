from google import genai
import time
import random
import os 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemma-3-27b-it"
NUM_ROUNDS = 1

client = genai.Client(api_key=API_KEY)

centuries = ["XII", "XVII", "XIX", "XXI", "XXV"]
countries = ["Italy", "Vietnam", "Argentina", "Morocco", "Canada"]

for i in range(1, NUM_ROUNDS + 1):
    century = random.choice(centuries)
    country = random.choice(countries)
    
    prompt = f"Write a very short but funny story of 4 or 5 sentences set in the {century} century in {country}."
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=1.0,
                top_p=0.95
            ),
        )
        
        print(f"--- story number {i} ---")
        print(f"context: {century} century in {country}")
        print(f"story: {response.text.strip()}")
        print(f"tokens: {response.usage_metadata.total_token_count}")

        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"error in request {i}: {e}")
    
    if i < NUM_ROUNDS:
        time.sleep(5)