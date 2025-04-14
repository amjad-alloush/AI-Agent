from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from agent import Agent


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize agent
agent = Agent()

# Load agent state if available
state_dir = os.getenv("AGENT_STATE_DIR", "agent_state")
if os.path.exists(state_dir):
    agent.load_state(state_dir)
   

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages."""
    data = request.json
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    
    # Process user input
    input_data = {
        'type': 'text',
        'content': user_input,
        'metadata': {
        'source': 'web',
        'user_id': data.get('user_id', 'anonymous')
        }
    }
    
    response = agent.process_input(input_data)
    
    # Save agent state periodically
    if os.path.exists(state_dir) or os.makedirs(state_dir, exist_ok=True):
        agent.save_state(state_dir)
        
    return jsonify({
        'message': response['response'],
        'actions': [action['tool'] for action in response.get('actions', [])]
        })

    if __name__ == '__main__':
        port = int(os.getenv("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False)