```python
import os
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("message", "")).strip()

    if not prompt:
        return jsonify({"error": "Please enter a question."}), 400

    if len(prompt) > 500:
        return jsonify({
            "error": "Keep your question under 500 characters."
        }), 400

    if not GEMINI_API_KEY:
        return jsonify({
            "error": "Gemini API key is not configured on the server."
        }), 500

    try:
        response = requests.post(
            GEMINI_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "system_instruction": {
                    "parts": [{
                        "text": (
                            "You are NovaMind AI, a helpful AI assistant. "
                            "Answer clearly and accurately. "
                            "Adapt explanations to the user's level."
                        )
                    }]
                },
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()
        answer = result["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify({"answer": answer})

    except requests.RequestException as exc:
        print("Gemini API error:", exc)
        return jsonify({
            "error": "Gemini API request failed. Check the server logs."
        }), 502

    except (KeyError, IndexError, ValueError) as exc:
        print("Invalid Gemini response:", exc)
        return jsonify({
            "error": "Gemini returned an unexpected response."
        }), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
```
