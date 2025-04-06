from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://127.0.0.1:5500"}})

# Load Hugging Face model
generator = pipeline("text-generation", model="distilgpt2")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    product = data.get("product")

    if not product:
        return jsonify({"error": "No product name provided"}), 400

    # Improved prompt
    prompt = (
        f"Create a 150-word SEO-optimized blog post about {product}. "
        f"Make sure to naturally include these keywords: 'best {product}', '{product} review', "
        f"'{product} 2025', and '{product} online'. Write in a friendly, informative tone and "
        f"structure it like a complete blog post with an engaging introduction and a clear conclusion."
    )

    result = generator(prompt, max_length=250, num_return_sequences=1)
    blog = result[0]["generated_text"]

    return jsonify({"blog": blog.strip()})

if __name__ == "__main__":
    app.run(debug=True)
