
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"


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

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are NovaMind AI, a helpful AI assistant. "
                            "Answer clearly, accurately, and adapt explanations "
                            "to beginner, intermediate, or advanced levels."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()
        answer = response.json()["message"]["content"].strip()

        return jsonify({"answer": answer})

    except requests.RequestException as exc:
        print("Ollama connection error:", exc)
        return jsonify({
            "error": (
                "Cannot connect to Ollama. Check that Ollama is running "
                "and the llama3 model is installed."
            )
        }), 503

    except (KeyError, ValueError) as exc:
        print("Invalid Ollama response:", exc)
        return jsonify({
            "error": "Ollama returned an unexpected response."
        }), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)