from flask import Flask, jsonify, render_template, request
from rag_system import RAGSystem
from agent_orchestrator import AgentOrchestrator
from teacher_agent import TeacherAgent
import json

app = Flask(__name__)

# Initialize systems
rag = RAGSystem(
    faiss_index_path="faiss_index.idx",
    vectors_path="vectors.pkl"
)

# Initialize orchestrator and teacher
orchestrator = AgentOrchestrator(rag)
teacher = TeacherAgent(orchestrator)

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page with chat interface."""
    try:
        if request.method == 'POST':
            message = request.form.get('message')
            mode = request.form.get('mode')
            print(f"Received request - Mode: {mode}, Message: {message}")  # Debug print
            
            if mode == 'quick_search':
                # Use RAG system directly for quick search
                result = rag.query(message)
                return jsonify({
                    'status': 'success',
                    'response': json.loads(result)
                })
            elif mode == 'learning':
                # Use teacher agent for learning mode
                print("Processing learning mode request")  # Debug print
                response = teacher.process_message(message)
                print(f"Teacher response: {response}")  # Debug print
                return jsonify({
                    'status': 'success',
                    'response': response
                })
            
        return render_template('index.html')
    except Exception as e:
        print(f"Error in route: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 