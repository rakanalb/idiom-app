# 🎯 Adam: Your Idioms Teaching Assistant

> An intelligent AI-powered application that makes learning English idioms fun and interactive. Featuring both quick search and personalized learning modes with Adam, your AI teaching companion.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.1.0-lightgrey.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-API-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🔍 Quick Search Mode
- Instantly find and understand English idioms
- Get clear explanations and real-world examples
- Perfect for quick reference and learning on the go

### 🎓 Interactive Learning Mode
Meet Adam, your personal AI teaching assistant that:
- Adapts to your English proficiency level
- Creates personalized learning paths
- Provides instant, constructive feedback
- Makes learning idioms engaging and fun

### 🚀 Advanced Technology
- **RAG System**: Combines FAISS similarity search with OpenAI embeddings
- **Conversational AI**: Natural, context-aware interactions
- **Modern UI**: Clean, responsive design that works everywhere

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI/ML**: OpenAI API, FAISS
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: FAISS Index, Pickle

## 📁 Project Structure
```
idiom-main/
│
├── app.py                  # Flask application server
├── agent_orchestrator.py   # AI agent coordination
├── teacher_agent.py       # Teaching AI logic
├── Data_preprocessing.py   # PDF processing & embeddings
├── rag_system.py          # RAG implementation
├── Procfile               # Heroku deployment config
├── render.yaml            # Render deployment config
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
├── idioms.pdf            # Source document for idioms
├── templates/
│   └── index.html        # Main UI template
├── README.md             # Project documentation
├── faiss_index.idx       # Generated FAISS index
└── vectors.pkl           # Generated embeddings
```
## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone & Navigate**
git clone <repository-url>
cd idiom-main

2. **Set Up Environment**
python -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate
pip install -r requirements.txt

3. **Configure Environment**
Create `.env` in root directory:
OPENAI_API_KEY=your_api_key_here

### Initial Setup

1. Add your idioms PDF to root directory
2. Run preprocessing:
python Data_preprocessing.py

### Launch Application
python app.py
Visit `http://localhost:5000` in your browser 🚀

## 🌐 Deployment

Ready for deployment on Render platform:

1. Create Render account
2. Set OPENAI_API_KEY in Render dashboard
3. Connect repository
4. Deploy!


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Special thanks to
 **Thamer Algahtani** ([@notthamer](https://github.com/notthamer)) for the collaborative development and innovative ideas

---

Made with ❤️ at Ironhack Barcelona