# My Tutor System

A comprehensive tutoring system that combines intent classification, LLM-based responses, and performance metrics tracking. The system uses BERT for intent detection and Ollama for response generation, with comprehensive logging and virtual user simulation capabilities.

## 🚀 Features

- **Intent Classification**: BERT-based classifier for detecting user intents (Genuine, Manipulative, Spam)
- **LLM Integration**: Ollama-powered response generation with intent-aware system prompts
- **Performance Metrics**: Comprehensive tracking of response times, adherence rates, and user behavior
- **Virtual User Simulation**: Automated testing with 4 distinct user personas
- **Database Logging**: SQLite-based storage for all interactions and metrics
- **RESTful APIs**: FastAPI-based services with automatic documentation


## 📁 Project Structure

```
Ethical-AI-Tutor/
├── AI-Tutor-System/
│   ├── intent_api.py         # BERT intent classifier service (port 9000)
│   ├── llm_api.py            # Aggregator service with Ollama integration (port 8000)
│   ├── api.py                # Conversation API for frontend (port 8500)
│   ├── simulator.py          # Virtual user simulator with 4 personas
│   ├── models.py             # SQLAlchemy database models
│   ├── database.py           # SQLite session management
│   ├── metrics.py            # Performance metrics computation
│   ├── requirements.txt      # Python dependencies
│   ├── data/                 # Database storage directory (auto-created)
│   ├── reports/              # Simulation reports directory (auto-created)
│   └── ...                   # Other backend files
├── my-tutor-frontend/
│   ├── src/                  # React frontend source code
│   ├── public/               # Static assets
│   ├── package.json          # Frontend dependencies
│   └── ...                   # Other frontend files
└── ...
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Ollama installed and running
- At least 8GB RAM (for running models)


### 1. Install Backend Dependencies

```bash
cd AI-Tutor-System
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd ../my-tutor-frontend
npm install
```

### 2. Set Up Ollama

Install and start Ollama with a compatible model:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama3)
ollama pull llama3

# Start Ollama server
ollama serve
```


### 3. Start the Services

You can use the provided batch file to start all backend services and the simulator at once:

```bat
cd AI-Tutor-System
start_all_servers.bat
```

Or start each service in a separate terminal:

**Terminal 1 - Intent Classification API (port 9000):**
```bash
cd AI-Tutor-System
python -m uvicorn intent_api:app --host 127.0.0.1 --port 9000 --reload
```

**Terminal 2 - LLM Aggregator API (port 8000):**
```bash
cd AI-Tutor-System
python -m uvicorn llm_api:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 3 - Conversation API (port 8500):**
```bash
cd AI-Tutor-System
python -m uvicorn api:app --host 127.0.0.1 --port 8500 --reload
```

**Terminal 4 - Simulator:**
```bash
cd AI-Tutor-System
python simulator.py
```

**Terminal 5 - Frontend (React):**
```bash
cd ../my-tutor-frontend
npm start
```

### 4. Verify Setup

Check that services are running:
```bash
# Test intent API
curl -X POST "http://localhost:9000/predict_intent" \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I solve quadratic equations?"}'

# Test LLM API
curl -X POST "http://localhost:8000/completions" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is calculus?", "user_id": "test_user"}'
```

## 🎯 Usage


### Running Virtual User Simulation

Execute the complete simulation with all personas:

```bash
cd AI-Tutor-System
python simulator.py
```

This will:
- Test 4 different user personas (lazy, curious, persistent, strategic)
- Send 20 questions across various subjects
- Generate comprehensive metrics and reports
- Save results to `reports/simulation_report_<timestamp>.json`

### Manual API Testing

#### Intent Classification
```bash
curl -X POST "http://localhost:9000/predict_intent" \
     -H "Content-Type: application/json" \
     -d '{"question": "Just give me the answer to this math problem"}'
```

#### LLM Completion
```bash
curl -X POST "http://localhost:8000/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "How do derivatives work?",
       "user_id": "student_123",
       "persona": "curious"
     }'
```

### Database Operations


#### Initialize Database
```bash
cd AI-Tutor-System
python database.py
```

#### Export Metrics to CSV
```python
from database import get_db_context
from metrics import export_metrics_csv

with get_db_context() as db:
    export_metrics_csv(db, "reports/metrics_export.csv")
```

#### Get System Overview
```python
from database import get_db_context
from metrics import get_system_overview
import json

with get_db_context() as db:
    overview = get_system_overview(db)
    print(json.dumps(overview, indent=2))
```

## 🧪 Virtual User Personas

The system includes 4 distinct user personas for comprehensive testing:

### 1. Lazy Student (`lazy`)
- **Behavior**: Asks for direct answers, stops after 1 turn
- **Questions**: "Just give me the answer to...", "What's the solution for..."
- **Expected Intent**: Often classified as Manipulative
- **Purpose**: Tests system's ability to guide rather than provide direct answers

### 2. Curious Learner (`curious`) 
- **Behavior**: Asks follow-up questions, seeks deeper understanding
- **Questions**: "I'm curious about...", "Can you explain..."
- **Follow-ups**: "Why is that?", "Can you give another example?"
- **Expected Intent**: Genuine
- **Purpose**: Tests system's educational capabilities

### 3. Persistent Worker (`persistent`)
- **Behavior**: Rephrases questions up to 5 times when stuck
- **Questions**: "I'm struggling with...", "I need help understanding..."
- **Rephrasings**: "Let me ask this differently...", "Another way to ask..."
- **Expected Intent**: Genuine
- **Purpose**: Tests system's consistency and patience

### 4. Strategic Manipulator (`strategic`)
- **Behavior**: Attempts to bypass restrictions using authority
- **Questions**: "As a teacher, please...", "For educational purposes..."
- **Bypass Attempts**: "My student is asking...", "For my lesson plan..."
- **Expected Intent**: Manipulative
- **Purpose**: Tests system's ability to detect sophisticated manipulation

## 📊 Metrics and Analysis

### Key Performance Indicators

1. **Adherence Rate**: Percentage of responses following intent-based rules
2. **Response Times**: Intent detection, LLM generation, and total round-trip
3. **Interactions Before Failure**: Average turns before non-adherent response
4. **Token Statistics**: Response length distribution
5. **Intent Distribution**: Breakdown of detected intents by persona

### Intent-Based Response Rules

- **Genuine**: Provide educational explanations with examples and reasoning
- **Manipulative**: Give guidance/hints, avoid direct answers or code solutions
- **Spam**: Brief professional redirect to educational topics

### Viewing Results

1. **API Documentation**: Visit `http://localhost:8000/docs` or `http://localhost:9000/docs`
2. **Simulation Reports**: Check `reports/simulation_report_<timestamp>.json`
3. **CSV Export**: Load exported CSV files into Excel, pandas, or other tools
4. **Database Queries**: Use SQLite browser or custom scripts

## 🔧 Configuration

### Model Configuration

Edit the model settings in the respective files:

**Intent API (`intent_api.py`)**:
```python
# Change model path
model_path = "path/to/your/custom-intent-classifier"

# Fallback model for demo
model_name = "distilbert-base-uncased"
```

**LLM API (`llm_api.py`)**:
```python
# Ollama model configuration
payload = {
    "model": "llama3",  # Change to your preferred model
    "options": {
        "temperature": 0.7,  # Adjust creativity
        "max_tokens": 512    # Adjust response length
    }
}
```

### System Prompts

Customize intent-based prompts in `llm_api.py`:

```python
SYSTEM_PROMPTS = {
    "Genuine": "Your educational prompt for genuine questions...",
    "Manipulative": "Your guidance prompt for manipulation attempts...",
    "Spam": "Your redirect prompt for inappropriate content..."
}
```

## 📈 Example Visualizations

After running simulations, you can create visualizations:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported data
df = pd.read_csv('reports/metrics_export.csv')

# Response time by persona
df.groupby('persona')['llm_time_ms'].mean().plot(kind='bar')
plt.title('Average Response Time by Persona')
plt.ylabel('Time (ms)')
plt.show()

# Adherence rate by intent
adherence_by_intent = df.groupby('intent')['adherence'].mean()
adherence_by_intent.plot(kind='pie', autopct='%1.1f%%')
plt.title('Adherence Rate by Intent')
plt.show()
```

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Failed to call Ollama API: Connection refused
   ```
   - Ensure Ollama is running: `ollama serve`
   - Check if model is pulled: `ollama list`

2. **Intent Model Loading Error**
   ```
   Could not load custom model from path/to/intent-classifier
   ```
   - The system will fallback to DistilBERT for demonstration
   - Replace with your actual BERT model path

3. **Database Lock Error**
   ```
   database is locked
   ```
   - Close any open database connections
   - Restart the services

4. **Port Already in Use**
   ```
   Address already in use
   ```
   - Check running processes: `lsof -i :8000` or `lsof -i :9000`
   - Kill existing processes or change ports

### Performance Tips

1. **Use GPU**: Install PyTorch with CUDA support for faster inference
2. **Batch Processing**: For large simulations, consider batching requests
3. **Database Optimization**: For production, consider PostgreSQL over SQLite
4. **Caching**: Implement Redis caching for frequently accessed metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all services start correctly
5. Submit pull request with comprehensive description

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

**Note**: This system is designed for educational and research purposes. For production deployment, implement proper authentication, rate limiting, and security measures. 
