from flask import Flask, jsonify, render_template, request, session
from rag_system import RAGSystem
from agent_orchestrator import AgentOrchestrator
from teacher_agent import TeacherAgent
import json
import uuid
import secrets

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
# Generate a secure random key
app.secret_key = secrets.token_hex(32)

# Initialize systems with error handling
try:
    rag = RAGSystem(
        faiss_index_path="faiss_index.idx",
        vectors_path="vectors.pkl"
    )
    orchestrator = AgentOrchestrator(rag)
    teacher = TeacherAgent(orchestrator)
except Exception as e:
    print(f"Error initializing systems: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page with chat interface."""
    try:
        # Ensure user has a session ID
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
            print(f"Created new session: {session['user_id']}")
            # Get initial greeting without requiring a message
            initial_response = teacher.get_initial_greeting(session['user_id'])
            return render_template('index.html', initial_message=initial_response)

        if request.method == 'POST':
            message = request.form.get('message', '').strip()
            mode = request.form.get('mode', '').strip()
            
            if not mode:
                return jsonify({
                    'status': 'error',
                    'error': 'Missing message or mode'
                }), 400
            
            print(f"Processing request - Session: {session['user_id']}, Mode: {mode}, Message: {message}")
            
            try:
                if mode == 'quick_search':
                    if not message:
                        return jsonify({
                            'status': 'error',
                            'error': 'Missing search query'
                        }), 400
                    result = rag.query(message)
                    return jsonify({
                        'status': 'success',
                        'response': json.loads(result)
                    })
                elif mode == 'learning':
                    if not message:
                        # Check if this is the first greeting or level question
                        if 'greeted' not in session:
                            session['greeted'] = True
                            response = teacher.get_initial_greeting(session['user_id'])
                        else:
                            response = teacher.get_level_question(session['user_id'])
                    else:
                        response = teacher.process_message(message, session['user_id'])
                    return jsonify({
                        'status': 'success',
                        'response': response
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'error': 'Invalid mode'
                    }), 400
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'error': 'Error processing your request'
                }), 500
            
        return render_template('index.html')
        
    except Exception as e:
        print(f"Unexpected error in route: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'An unexpected error occurred'
        }), 500

@app.errorhandler(Exception)
def handle_error(e):
    print(f"Unhandled error: {str(e)}")
    return jsonify({
        'status': 'error',
        'error': 'An unexpected error occurred'
    }), 500 

if __name__ == '__main__':
    app.run(debug=True) 