# RAG-based Chatbot with Llama2 and MongoDB

This project implements a RAG (Retrieval-Augmented Generation) based chatbot using Llama2 as the language model, MongoDB for data and session management, and a document ingestion system for processing PDF documents and storing the results in MongoDB. The chatbot supports personalized user sessions and history, including user registration, login, and chat history management.

Using a GPU environment is recommended for this project


## Architecture
![Screenshot from 2024-07-05 13-59-06](https://github.com/NivedKris/RAG-based-chatbot-using-llama2-and-mongodb/assets/100478612/31b9502b-e4bb-45dc-9188-3f1edb4fb5a7)

## Features

### Chatbot Capabilities
- **Large Language Model**: Utilizes Llama2 running on a local machine via Ollama for generating responses.
- **Session Management**: Manages user sessions and chat history using MongoDB.
- **User Management**: Supports user registration and login.
- **Personalized Responses**: Provides personalized responses based on user history and previous interactions.

### Document Ingestion System
- **PDF Processing**: Extracts text from PDF documents and processes them into structured chunks.
- **Chunking and Metadata**: Splits documents into chunks and generates summaries and titles using Llama2.
- **Embedding**: Generates embeddings for each chunk using Text2vecEmbeddings.
- **MongoDB Storage**: Stores the processed chunks, their metadata, and embeddings in MongoDB.

## Setup and Installation

### Prerequisites
- Python 3.8+
- MongoDB
- Necessary Python packages (see `requirements.txt`)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Setting Up MongoDB
To handle the requirements for equipping the RAG application with the capabilities of storing interaction or conversation history, two new collections must be created alongside the collection that will hold the main application data.

1. **Register a free Atlas account** or sign in to your existing Atlas account.
2. **Deploy your first cluster** following the instructions (select Atlas UI as the procedure).
3. **Create the database:** `langchain_chatbot`.
4. **Within the database `langchain_chatbot`, create the following collections:**
   - `data`: Holds all data that acts as a knowledge source for the chatbot.
   - `history`: Holds all conversations held between the chatbot and the application user.
5. **Create a vector search index named `vector_index` for the `data` collection:**
   This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index:
   ```json
   {
     "fields": [
       {
         "numDimensions": 786,
         "path": "embedding",
         "similarity": "cosine",
         "type": "vector"
       }
     ]
   }
   ```
6. **Obtain the connection URI string** to the created Atlas cluster to establish a connection between the databases and the current development environment. Follow the steps to get the connection string from the Atlas UI.
   ```python
   MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")
   ```

### Setting Up Ollama for Llama2
1. **Download Ollama:**
   - Visit the [Ollama website](https://www.ollama.com) and download the appropriate version for your operating system.
2. **Install Ollama:**
   - Follow the installation instructions provided on the Ollama website.
2. **Install llama2:**
     ```bash
   ollama run llama2
   ```
    -This will automatically install llama2 in you machine 
     
3. **Start the Llama2 server:**
   ```bash
   ollama start
   ```

### Configuration
- Update MongoDB connection URI and database details in the script.
- Adjust folder paths for document ingestion as needed.

## Usage

### Running the Chatbot
1. **Start the Llama2 server:**
   ```bash
   ollama start
   ```

2. **Run the chatbot:**
   ```bash
   python main.py
   ```

### Document Ingestion
1. **Prepare folders for document ingestion:**
   ```bash
   mkdir input_pdfs processing_pdfs processed_pdfs
   ```

2. **Start monitoring and processing documents:**
   ```bash
   python upload_documents.py
   ```




## Contributions

Any contribution or further development to the architecture is highly appreciated

Email:-[nithupd@gmail.com](mailto:neerajroythomas@gmail.com)
