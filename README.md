# DocuMind — AI-Powered Document Intelligence & RAG Platform

DocuMind is an AI-powered document question-answering platform that allows users to upload PDF documents and ask questions in natural language.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-based answers using Google's Gemini API.

## Features

- Upload and process multiple PDF documents.
- Extract text and divide it into smaller chunks.
- Retrieve relevant information using TF-IDF and cosine similarity.
- Generate answers using Google Gemini.
- Display source document names and page references.
- Handle questions when the uploaded documents do not contain sufficient information.
- Interactive web interface built with Streamlit.

## Tech Stack

- **Python** — Core programming language
- **Streamlit** — User interface
- **PyPDF** — PDF text extraction
- **Scikit-learn** — TF-IDF and cosine similarity
- **Google Gemini API** — AI-generated responses
- **Python-dotenv** — Environment variable management

## Project Structure

```text
DocuMind_RAG/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuMind_RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your Gemini API key

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Never upload your `.env` file to GitHub.

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How It Works

1. User uploads one or more PDF documents.
2. DocuMind extracts text and splits it into smaller chunks.
3. TF-IDF converts the text chunks into numerical representations.
4. Cosine similarity identifies the most relevant chunks for a question.
5. The retrieved context is sent to Gemini.
6. Gemini generates an answer grounded in the retrieved document content.
7. The application displays the answer along with source references.

## Security

- API keys are stored in environment variables.
- The `.env` file is excluded from Git using `.gitignore`.
- Users must provide their own Gemini API key.

## Future Improvements

- Semantic search using embedding models.
- Support for additional document formats.
- Improved retrieval and ranking.
- Conversation history and memory.
- Deployment for public access.

---

**DocuMind** — Making document understanding smarter and easier.