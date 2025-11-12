"""
Flask server for DIA - Dementia Information Assistant web interface
"""
import sys
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dia_agent import DIAAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set UTF-8 encoding for stdout/stderr to handle emojis on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)
CORS(app)

# Initialize DIA agent
agent = None

def get_agent():
    """Get or initialize the DIA agent"""
    global agent
    if agent is None:
        agent = DIAAgent()
    return agent

@app.route('/')
def index():
    """Serve the main HTML page with injected base URL"""
    base_url = os.getenv('BASE_URL', 'http://localhost:8080')
    
    # Read the HTML file
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Inject base URL as a JavaScript variable before the closing </head> tag
    script_tag = f'<script>const BASE_URL = "{base_url}";</script>'
    html_content = html_content.replace('</head>', f'{script_tag}\n</head>')
    
    from flask import Response
    return Response(html_content, mimetype='text/html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        # Get agent instance
        dia_agent = get_agent()

        # Get response from agent
        response = dia_agent.chat(user_message)

        # Parse follow-up questions from response
        follow_up_questions = []
        if '**You might also want to know:**' in response:
            parts = response.split('**You might also want to know:**')
            main_response = parts[0].strip()

            # Extract questions
            questions_text = parts[1].strip()
            lines = questions_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    question = line.split('.', 1)[1].strip()
                    follow_up_questions.append(question)
        else:
            main_response = response

        return jsonify({
            'response': main_response,
            'follow_up_questions': follow_up_questions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        dia_agent = get_agent()
        dia_agent.clear_history()
        return jsonify({'success': True, 'message': 'Conversation history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/welcome', methods=['GET'])
def get_welcome():
    """Get welcome message from the bot"""
    try:
        dia_agent = get_agent()
        welcome_message = dia_agent.get_welcome_message()
        
        # Parse follow-up questions from welcome message
        follow_up_questions = []
        if '**You might also want to know:**' in welcome_message:
            parts = welcome_message.split('**You might also want to know:**')
            main_message = parts[0].strip()

            # Extract questions
            questions_text = parts[1].strip()
            lines = questions_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    question = line.split('.', 1)[1].strip()
                    follow_up_questions.append(question)
        else:
            main_message = welcome_message

        return jsonify({
            'welcome_message': main_message if '**You might also want to know:**' in welcome_message else welcome_message,
            'follow_up_questions': follow_up_questions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        print("\n" + "="*60)
        print("  Starting DIA Web Server...")
        print("="*60)
        print("\nInitializing DIA Agent...")

        # Initialize agent on startup to check for errors
        get_agent()

        base_url = os.getenv('BASE_URL', 'http://localhost:8080')
        print("\nServer ready!")
        print("\nOpen your browser and go to:")
        print(f"  {base_url}")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")

        app.run(debug=True, host='0.0.0.0', port=8080)
    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Created a .env file with:")
        print("   - GOOGLE_CLOUD_PROJECT (your GCP project ID)")
        print("   - GOOGLE_CLOUD_LOCATION (e.g., asia-south1)")
        print("   - RAG_CORPUS (your RAG corpus path)")
        print("2. Set up Google Cloud authentication")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
