# DIA Web Interface - Quick Start Checklist

## Before Your Demo Tomorrow

### 1. Open Terminal
- Open Git Bash or Command Prompt
- Navigate to project folder

### 2. Run These Commands

```bash
# Navigate to project
cd "E:\Main\Desktop\Dementia Awareness Datasets\for drop down"

# Authenticate with Google Cloud (do this first!)
gcloud auth application-default login
gcloud auth application-default set-quota-project tech-bharath

# Start the server
python server.py
```

### 3. Test It
- Open browser: **http://localhost:8080**
- Test a question: "What are the early signs of dementia?"
- Make sure it works before your demo!

### 4. During Demo
- Keep terminal window visible
- Have sample questions ready
- Show the clickable follow-up questions feature

---

## If Something Goes Wrong

### Flask Not Found
```bash
pip install flask flask-cors
```

### Authentication Error
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project tech-bharath
# Then restart server (Ctrl+C, then python server.py)
```

### Server Won't Start
- Check if another server is running on port 8080
- Close and restart terminal
- Try again

---

## Sample Questions for Demo

1. "What are the early signs of dementia?"
2. "How can I support a family member with dementia?"
3. "What is the difference between Alzheimer's and dementia?"
4. "What lifestyle changes can reduce dementia risk?"
5. "What are the stages of dementia?"

---

## Important URLs

- Web Interface: http://localhost:8080
- Alternative: http://127.0.0.1:8080

## Important Files

- `server.py` - The web server (Flask backend)
- `index.html` - The web interface (frontend)
- `dia_agent.py` - The AI agent with RAG
- `.env` - Your Google Cloud credentials

---

**Ready?** Just run `python server.py` and open http://localhost:8080!
