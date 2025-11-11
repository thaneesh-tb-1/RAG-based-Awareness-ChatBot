# DIA - Dementia Information Assistant

A CLI-based chatbot powered by Google Gemini AI with RAG (Retrieval-Augmented Generation) capabilities, designed to educate adults (40+) and caregivers about dementia in a calm, clear, and friendly way.

## Quick Start (Web Interface)

**To run the web interface tomorrow:**

```bash
# 1. Navigate to project
cd "E:\Main\Desktop\Dementia Awareness Datasets\for drop down"

# 2. Authenticate with Google Cloud (if needed)
gcloud auth application-default login
gcloud auth application-default set-quota-project tech-bharath

# 3. Start the web server
python server.py

# 4. Open browser to: http://localhost:5000
```

**That's it!** The web interface will be running on http://localhost:5000

*Note: If you get a "Flask not found" error, run: `pip install flask flask-cors`*

---

## Features

- **RAG-Powered Responses**: Uses Vertex AI RAG to retrieve information from trusted health sources (WHO, Alzheimer's Association, NIMHANS, Dementia India Alliance, NHS)
- **Smart Context Caching**: Automatically caches retrieved information and reuses it for related follow-up questions, reducing RAG queries and improving response time
- **Follow-up Question Suggestions**: Every response includes 2-3 relevant follow-up questions to guide your learning journey
- **Conversational Interface**: Interactive CLI chat interface with conversation history
- **Streaming Responses**: Real-time response streaming for a natural chat experience
- **Safety-First**: Only provides verified information, never diagnoses
- **Warm & Clear Tone**: Uses simple Indian-English with psychoeducational approach

## Prerequisites

- Python 3.12 or higher
- Google Cloud Platform account with Vertex AI access
- Google Cloud API Key
- Access to RAG corpus in Vertex AI

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd "E:\Main\Desktop\Dementia Awareness Datasets\for drop down"
   ```

2. **Install dependencies using uv (recommended) or pip**

   Using uv:
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

3. **Configure environment variables**

   The `.env` file is already set up with your configuration. Ensure it contains:
   ```
   GOOGLE_CLOUD_API_KEY=your_actual_api_key
   GOOGLE_CLOUD_PROJECT=tech-bharath
   GOOGLE_CLOUD_LOCATION=asia-south1
   RAG_CORPUS=projects/tech-bharath/locations/asia-south1/ragCorpora/288230376151711744
   ```

   **Note**: Your API key is already configured in the `.env` file.

## Usage

### Running DIA CLI

You can run DIA in the command line in several ways:

**Option 1: Using Python directly**
```bash
python main.py
```

**Option 2: Using the dia_agent module**
```bash
python dia_agent.py
```

**Option 3: Using the installed command (after installation)**
```bash
dia
```

### Running DIA Web Interface

DIA includes a modern web interface with a beautiful gradient UI that you can run in your browser.

#### Prerequisites for Web Interface

1. **Google Cloud Authentication** (Required)
   - You must authenticate with Google Cloud before running the web server
   - This is required for accessing Vertex AI and RAG corpus

#### Step-by-Step Instructions

**Step 1: Navigate to the project directory**
```bash
cd "E:\Main\Desktop\Dementia Awareness Datasets\for drop down"
```

**Step 2: Install web server dependencies**
```bash
pip install flask flask-cors
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

**Step 3: Authenticate with Google Cloud**

This is a **critical step** - the server will not work without authentication!

```bash
gcloud auth application-default login
```

This will:
- Open your browser for authentication
- Save credentials to your local machine
- Allow the DIA agent to access Vertex AI

**Step 4: Set quota project (recommended)**
```bash
gcloud auth application-default set-quota-project tech-bharath
```

This prevents quota-related errors.

**Step 5: Start the web server**
```bash
python server.py
```

You should see:
```
============================================================
  Starting DIA Web Server...
============================================================

Initializing DIA Agent...

Server ready!

Open your browser and go to:
  http://localhost:5000

Press Ctrl+C to stop the server
============================================================
```

**Step 6: Open your browser**

Navigate to: **http://localhost:5000**

**Step 7: Start chatting!**

Type your question in the input box and click "Send" or press Enter.

#### Web Interface Features

- **Modern Gradient UI**: Beautiful purple gradient design
- **Responsive Chat Interface**: Works on desktop and mobile
- **Message Bubbles**: Distinct user and bot messages with avatars
- **Clickable Follow-up Questions**: Click suggested questions to ask them instantly
- **Loading Indicators**: Animated dots while waiting for responses
- **Clear Button**: Reset conversation history anytime
- **Real-time Streaming**: See responses as they're generated
- **Emoji Support**: Full UTF-8 support for emojis in responses

#### Stopping the Server

To stop the web server, press `Ctrl+C` in the terminal where the server is running.

#### Tips for Presenting/Demoing

When showing this to others at work:

1. **Start the server in advance** (5 minutes before your demo)
   - Verify it's working by testing a question yourself
   - Keep the terminal window visible to show real-time processing

2. **Prepare sample questions**:
   - "What are the early signs of dementia?"
   - "How can I support a family member with dementia?"
   - "What is the difference between Alzheimer's and dementia?"
   - "What lifestyle changes can reduce dementia risk?"

3. **Demonstrate key features**:
   - Show how responses are generated in real-time
   - Click on follow-up questions to demonstrate interactivity
   - Use the "Clear" button to start fresh conversations
   - Highlight the clean, user-friendly interface

4. **Keep terminal visible**: The server terminal shows useful debug info

5. **Internet required**: Make sure you have stable internet (needs GCP access)

### Chat Commands

Once DIA is running, you can use these commands:

- **Ask questions**: Simply type your question about dementia
  ```
> You: What are the early signs of dementia?
  ```

- **Clear conversation**: Start a fresh conversation
  ```
> You: clear
  ```

- **Exit**: Leave the chat
  ```
> You: exit
  ```
  or `quit` or press `Ctrl+C`

## How Smart Caching Works

DIA now features an intelligent caching system that improves performance and provides a better user experience:

1. **First Query**: When you ask a question, DIA retrieves relevant information from the RAG corpus
2. **Caching**: The response and context are automatically cached
3. **Follow-up Questions**: When you ask a related follow-up question, DIA:
   - Compares your question with cached queries using keyword similarity (Jaccard similarity)
   - If similarity exceeds 30%, it uses the cached context instead of querying RAG again
   - You'll see a "💾 [Using cached context - no RAG query needed]" indicator
4. **Cache Management**: The cache stores the last 10 queries to balance performance and memory

This means faster responses for related questions and more efficient use of the RAG system!

## Example Conversation

```
============================================================
  Welcome to DIA - Dementia Information Assistant
============================================================

I'm here to help you learn about dementia in a friendly,
clear way. Ask me anything about dementia awareness,
symptoms, care, or support.

Commands:
  - Type your question to chat
  - Type 'clear' to start a new conversation
  - Type 'exit' or 'quit' to leave

============================================================

>� You: What is dementia?

> DIA: Dementia is a syndrome where brain function gradually
declines, affecting memory, thinking, and daily activities.
It's not a normal part of aging, though it's more common in
older adults. There are different types like Alzheimer's
disease, vascular dementia, and others, each affecting the
brain differently. Early detection and proper support can
help maintain quality of life.
```

## Project Structure

```
.
|-- dia_agent.py          # Main DIA agent implementation
|-- main.py               # Entry point
|-- .env                  # Environment variables (your API key)
|-- .env.example          # Template for environment variables
|-- pyproject.toml        # Project configuration and dependencies
|-- .gitignore            # Git ignore rules (includes .env)
`-- README.md             # This file
```

## Configuration

### System Prompt

DIA is configured with a specific system prompt that ensures:
- Warm, simple Indian-English communication
- Stigma-free, psychoeducational approach
- Use of verified information only
- Appropriate structure and comprehensiveness
- Human-like chat experience

### Model Settings

- **Model**: `gemini-2.5-flash-preview-09-2025`
- **Temperature**: 1 (for natural responses)
- **Top-p**: 0.95
- **Max Output Tokens**: 65535
- **Safety Settings**: All categories set to OFF (for medical education context)

## Security Note

The `.env` file is excluded from git tracking to protect your API key. Never commit your actual API key to version control.

## Troubleshooting

### CLI Issues

#### "GOOGLE_CLOUD_API_KEY not found"
- Ensure your `.env` file exists and contains the API key
- Check that the `.env` file is in the same directory as the scripts

#### "RAG_CORPUS not found"
- Verify your `.env` file has the correct RAG corpus path
- Ensure you have access to the Vertex AI RAG corpus

#### Import Errors
- Run `uv sync` or `pip install -e .` to install dependencies
- Ensure you're using Python 3.12 or higher

#### API Authentication Errors
- Verify your Google Cloud API key is valid
- Check that your GCP project has Vertex AI API enabled
- Ensure you have proper permissions for the RAG corpus

### Web Interface Issues

#### "ModuleNotFoundError: No module named 'flask'"
**Problem**: Flask is not installed

**Solution**:
```bash
pip install flask flask-cors
```

#### "Error generating response: Reauthentication is needed"
**Problem**: Google Cloud credentials have expired or are not set up

**Solution**:
1. Run authentication command:
   ```bash
   gcloud auth application-default login
   ```
2. Set quota project:
   ```bash
   gcloud auth application-default set-quota-project tech-bharath
   ```
3. **Restart the server** (critical step!):
   - Press `Ctrl+C` to stop the server
   - Run `python server.py` again

The server must be restarted after authentication for changes to take effect.

#### "Error: 'charmap' codec can't encode character"
**Problem**: Windows console encoding issue with emojis

**Solution**: This has been fixed in the latest version of `server.py`. If you still see this error:
1. Make sure you have the latest `server.py` file
2. The file should have UTF-8 encoding setup at the top:
   ```python
   if sys.platform == 'win32':
       sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
   ```

#### "Cannot connect to server" in browser
**Problem**: Server is not running or wrong URL

**Solution**:
1. Check that the server is running (you should see output in terminal)
2. Make sure you're using the correct URL: `http://localhost:5000`
3. Check for firewall blocking port 5000
4. Try alternative URL: `http://127.0.0.1:5000`

#### Server starts but responses don't work
**Problem**: Usually authentication or environment variable issues

**Solution**:
1. Check that `.env` file exists in the project directory
2. Verify Google Cloud authentication (Step 3 in web interface setup)
3. Check server terminal for error messages
4. Restart the server after fixing any issues

#### Follow-up questions not appearing
**Problem**: Response parsing issue or API response format

**Solution**:
- This is normal if the agent doesn't generate follow-up questions
- The agent should generate follow-up questions according to the system prompt
- Check if the main response is working correctly
- If consistently missing, check the DIA agent configuration

### Quick Restart Procedure (if server has issues)

If the web server is acting strange or giving errors:

1. **Stop the server**: Press `Ctrl+C` in the terminal
2. **Verify authentication** (if you see auth errors):
   ```bash
   gcloud auth application-default login
   gcloud auth application-default set-quota-project tech-bharath
   ```
3. **Restart the server**:
   ```bash
   python server.py
   ```
4. **Refresh your browser**: Go to `http://localhost:5000`

## Support

For issues or questions about DIA, please check:
- Your environment configuration
- Google Cloud Platform permissions
- API key validity

## License

This project is designed for educational and awareness purposes about dementia.
