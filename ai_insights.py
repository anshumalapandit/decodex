import os
import requests
from dotenv import load_dotenv


load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")


def generate_structural_insight(summary_text):
    """
    Send structured summary to Grok and return interpretation.
    """

    if not GROK_API_KEY:
        return "Grok API key not found."

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are a regulatory market surveillance analyst."},
            {"role": "user", "content": summary_text}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        # Check for HTTP errors
        if response.status_code != 200:
            return f"Grok API Error {response.status_code}: {response.text}"

        result = response.json()
        
        # Check if response has expected structure
        if "choices" not in result:
            return f"Unexpected API response format: {result}"
        
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling Grok API: {str(e)}"