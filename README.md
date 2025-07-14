This project is a conversational AI agent that uses your own documents (PDFs, text files, etc.) as knowledge sources and OpenAI’s powerful LLMs to answer questions with context-sensitive, accurate responses.
🚀 Getting Started

Clone the repo

git clone https://github.com/yourusername/Talk-With-Pdf.git
cd Talk-With-Pdf

Create a virtual environment

python3 -m venv venv       # or `python -m venv venv` on Windows
source venv/bin/activate   # or `venv\Scripts\activate` on Windows

Install dependencies

pip install -r requirements.txt

Set your OpenAI API Key

export OPENAI_API_KEY="your-secret-key"

Run the Chatbot

streamlit run app.py

This will launch a local server (usually at http://localhost:8501) where you can chat with your data.

