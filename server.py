"""
Flask server for DIA - Dementia Information Assistant web interface
"""
import sys
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dia_agent import DIAAgent
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000').rstrip('/')
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')

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
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    # Inject base URL configuration into HTML
    # Replace the fallback empty string with the actual BASE_URL from .env
    html_content = html_content.replace(
        "const API_BASE_URL = window.API_BASE_URL || '';",
        f'const API_BASE_URL = window.API_BASE_URL || "{BASE_URL}";'
    )
    return html_content

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

        # Debug: Print full response to verify citations are included
        print(f"\n[DEBUG - Full Response Length: {len(response)} chars]")
        if "References" in response:
            print("[DEBUG - ✓ References section found in response]")
            # Show a snippet of the references
            ref_start = response.find("References")
            snippet = response[ref_start:ref_start+200] if ref_start != -1 else ""
            print(f"[DEBUG - References snippet: {snippet[:150]}...]")
        else:
            print("[DEBUG - ✗ No References section found in response]")

        # Parse follow-up questions from response
        follow_up_questions = []
        main_response = response
        
        if '**You might also want to know:**' in response:
            parts = response.split('**You might also want to know:**', 1)
            main_response = parts[0].strip()

            # Extract questions from the second part
            if len(parts) > 1:
                questions_text = parts[1].strip()
                lines = questions_text.split('\n')
                
                print(f"[DEBUG - Parsing follow-up questions from {len(lines)} lines]")
                
                for line in lines:
                    line = line.strip()
                    # More flexible parsing: handle "1.", "2.", "3." or numbered lists
                    if line:
                        # Try to match numbered list items (1., 2., 3., etc.)
                        match = re.match(r'^(\d+)\.\s*(.+)$', line)
                        if match:
                            question = match.group(2).strip()
                            if question:
                                follow_up_questions.append(question)
                                print(f"[DEBUG - Found follow-up question: {question[:50]}...]")
                        # Also handle bullet points or dashes
                        elif line.startswith('- ') or line.startswith('• '):
                            question = line.lstrip('- •').strip()
                            if question:
                                follow_up_questions.append(question)
                                print(f"[DEBUG - Found follow-up question (bullet): {question[:50]}...]")
                
                print(f"[DEBUG - Extracted {len(follow_up_questions)} follow-up questions]")
        else:
            print("[DEBUG - No 'You might also want to know' section found in response]")

        # Debug: Verify main_response includes references
        if "References" in main_response:
            print("[DEBUG - ✓ References included in main_response being sent to frontend]")
        else:
            print("[DEBUG - ✗ References NOT in main_response - may have been stripped]")
        
        # Debug: Print follow-up questions count
        print(f"[DEBUG - Sending {len(follow_up_questions)} follow-up questions to frontend]")

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
            'welcome_message': main_message,
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

        print("\nServer ready!")
        print("\nOpen your browser and go to:")
        print(f"  {BASE_URL}")
        print(f"\nServer running on {HOST}:{PORT}")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")

        app.run(debug=True, host=HOST, port=PORT)
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
