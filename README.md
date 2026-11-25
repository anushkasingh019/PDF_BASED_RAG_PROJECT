# 📚 PDF-Based RAG Chatbot

A simple AI-powered chatbot that allows users to upload PDF documents and ask questions based on their content. The chatbot uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded PDF and generate meaningful answers.

Instead of searching through an entire PDF manually, users can simply ask questions and get answers based on the document.

## 🚀 Features

- **PDF Upload:** Upload PDF documents directly through the web interface.
- **Document Processing:** Extracts and processes text from uploaded PDFs.
- **Smart Retrieval:** Finds relevant information according to the user's question.
- **AI-Powered Answers:** Generates context-based responses using an LLM.
- **Interactive Chat Interface:** Ask multiple questions about the uploaded document.
- **Easy to Use:** Simple and user-friendly Streamlit interface.

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Community
- Vector Database
- Embeddings
- Large Language Model (LLM)
- PyPDF

## ⚙️ How It Works

The project follows the Retrieval-Augmented Generation (RAG) approach.

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** Text is extracted from the PDF.
3. **Text Splitting:** The extracted text is divided into smaller chunks.
4. **Embedding Generation:** Embeddings are created from the text chunks.
5. **Vector Storage:** The embeddings are stored in a vector database.
6. **Question Processing:** When the user asks a question, the system searches for relevant chunks.
7. **Answer Generation:** The retrieved information is passed to the language model to generate an answer.

## 📁 Project Structure

```text
RAG-Chatbot/
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd RAG-Chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure API Key

Create a `.env` file in the project folder and add your API key:

```env
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual API key.

### 7. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser.

## 🎯 Use Cases

- Understanding academic notes and study materials.
- Asking questions from lecture PDFs.
- Extracting specific information from documents.
- Quickly revising topics without reading the entire PDF.
- Searching information from lengthy documents.

## 🔮 Future Improvements

- Support for multiple PDF uploads.
- Chat history and conversation memory.
- Source citations with page numbers.
- Support for different document formats.
- Improved document retrieval and response accuracy.

## 👩‍💻 Author

**Anushka Singh**

B.Tech Computer Science and Engineering Student

GitHub: [anushkasingh019](https://github.com/anushkasingh019)

---

⭐ If you find this project useful, consider giving it a star!