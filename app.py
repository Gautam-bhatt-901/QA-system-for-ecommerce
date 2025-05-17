
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import re
import helper
from flask import session
import os
encoder_model = helper.encoder_model
qa_pipeline = helper.qa_pipeline

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'development-secret-key')
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("product_url")
        if url:
            data = helper.scrape_product_data(url)
            if data:
                session["product_data"] = data
                return redirect(url_for("chat"))
            else:
                return render_template_string(PAGE_TEMPLATE, error="⚠️ Failed to scrape product info.", stage="input")
        else:
            return render_template_string(PAGE_TEMPLATE, error="⚠️ Enter a valid URL.", stage="input")

    return render_template_string(PAGE_TEMPLATE, stage="input")


@app.route('/chat', methods = ["GET","POST"])
def chat():
    if request.method == "POST":
        documents = session.get("product_data")
        data = request.get_json(force=True) 
        question = data.get("question", "").strip()
        # Search for relevant documents
        query_embedding, index, documents = helper.generate_embedding(documents=documents, encoder_model=encoder_model,question = question)
        distances, indices = index.search(query_embedding.astype('float32'), k = 3)
        # Get context from the most relevant document
        context = documents[indices[0][0]]
        # Get answer from BERT
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"] if result["score"] > 0.1 else "I don't know."
        return jsonify({"answer":answer})
    else:
        return render_template_string(PAGE_TEMPLATE, stage="chat")


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🛒 E-Commerce Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; padding: 40px; }
        .container { max-width: 700px; margin: auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h2 { margin-top: 0; }
        input[type="text"] { width: 75%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
        button { padding: 10px 15px; border: none; border-radius: 5px; background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .chat-log { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        .user { text-align: right; margin-bottom: 10px; }
        .bot { text-align: left; margin-bottom: 10px; }
        .error { color: red; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        {% if stage == 'input' %}
            <h2>🔗 Enter Product URL</h2>
            <form method="POST">
                <input type="text" name="product_url" placeholder="Paste Flipkart product URL..." required />
                <button type="submit">Load Product</button>
            </form>
            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}
        {% elif stage == 'chat' %}
            <h2>🤖 Ask About the Product</h2>
            <div class="chat-log" id="chat-log"></div>
            <input type="text" id="user-input" placeholder="Ask a question..." />
            <button onclick="sendMessage()">Send</button>
        {% endif %}
    </div>

    {% if stage == 'chat' %}
    <script>
        async function sendMessage() {
            const input = document.getElementById("user-input");
            const chatLog = document.getElementById("chat-log");
            const question = input.value.trim();
            if (!question) return;

            chatLog.innerHTML += `<div class="user"><strong>You:</strong> ${question}</div>`;
            input.value = "";

            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });

            const data = await res.json();
            chatLog.innerHTML += `<div class="bot"><strong>Bot:</strong> ${data.answer}</div>`;
            chatLog.scrollTop = chatLog.scrollHeight;
        }
    </script>
    {% endif %}
</body>
</html>
"""

if __name__ ==  "__main__":
    app.run()