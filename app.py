from flask import Flask, jsonify, render_template, request
import json
from rag_system import RAGSystem, save_response_to_json

app = Flask(__name__)

# Cache the RAG system instance (will use environment variable for API key)
rag_system = RAGSystem(
    faiss_index_path="faiss_index.idx",
    vectors_path="vectors.pkl"
)

def load_idioms():
    """Load idioms from the JSON file."""
    try:
        with open('idioms_response.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No idioms found"}

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page with form and results."""
    try:
        if request.method == 'POST':
            question = request.form['question']
            print(f"Received question: {question}")
            
            # Use cached RAG system
            response = rag_system.query(question)
            save_response_to_json(response)
            
            # Load and return results
            idioms = load_idioms()
            return render_template('index.html', idioms=idioms, question=question)
            
        return render_template('index.html')
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', error=str(e))

@app.route('/idioms', methods=['GET'])
def get_idioms():
    """Return all idioms."""
    return jsonify(load_idioms())

@app.route('/idioms/<int:number>', methods=['GET'])
def get_idiom(number):
    """Return a specific idiom by number."""
    idioms = load_idioms()
    for idiom in idioms.get('idioms', []):
        if idiom['number'] == number:
            return jsonify(idiom)
    return jsonify({"error": "Idiom not found"}), 404

if __name__ == '__main__':
    app.run(debug=True) 