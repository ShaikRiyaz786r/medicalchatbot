import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
load_dotenv()

app = Flask(__name__)

try:
    genai.configure(api_key="")
    pc = Pinecone(api_key='')
except (KeyError, TypeError) as e:
    print("FATAL ERROR: Ensure GOOGLE_API_KEY and PINECONE_API_KEY are set in your .env file.")
    exit()

embedding_model = "models/text-embedding-004"
llm = genai.GenerativeModel("gemini-1.5-flash-latest")
index_name = "medical-chatbot2"

if index_name in pc.list_indexes().names():
    index = pc.Index(index_name)
    print("Clients and models initialized successfully.")
else:
    index = None
    print(f"FATAL ERROR: Pinecone index '{index_name}' does not exist.")


# --- RAG FUNCTION ---
def get_rag_answer(query: str):
    if not index:
        return "Pinecone index is not available. Please run the ingest_data.py script first."
        
    # --- RETRIEVE ---
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    
    # --- AUGMENT ---
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n---\n"
    
    prompt = f"""
    You are a friendly and helpful medical assistant. Your primary goal is to answer questions based on the provided context.

    Follow these rules:
    1. If the user's question is a simple greeting, a thank you, or casual small talk (like 'hi', 'hello', 'how are you?', 'thanks'), respond politely in a conversational way. For these cases, IGNORE the context.
    2. If the user's question is about a specific medical or scientific topic, you MUST base your answer strictly on the provided 'CONTEXT'.
    3. If the question is about a specific topic but the 'CONTEXT' does not contain the necessary information to answer it, you MUST respond with: "Based on the provided information, I cannot answer that question."

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """

    # --- GENERATE ---
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error during generation: {e}")
        return "Sorry, an error occurred while generating the answer."

# --- FLASK ROUTES ---
@app.route('/')
def home():
    print('first request')
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    print('inside ask')
    data = request.get_json()
    user_question = data.get('question')
    
    if not user_question:
        return jsonify({"error": "No question provided."}), 400
    
    bot_answer = get_rag_answer(user_question)
    return jsonify({"answer": bot_answer})

# --- RUN APP ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)