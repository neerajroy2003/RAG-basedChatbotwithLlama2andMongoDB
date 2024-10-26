import uuid
import os
import time
import shutil
from io import BytesIO
import PyPDF2
from pymongo import MongoClient
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import subprocess
import requests
import json
from langchain_community.embeddings.text2vec import Text2vecEmbeddings


# MongoDB connection URI and client setup
MONGODB_URI = "mongodb+srv://markorganisation:bS7HUOnSsADbp8lY@chat.xgrad5y.mongodb.net/?retryWrites=true&w=majority&appName=chat"

client = MongoClient(MONGODB_URI)

db = client['langchain_chatbot'] # Database name
collection = db['data'] # Collection name

# Create a search index on the "document_id" field
collection.create_index([("document_id", 1)])

# Function to set executable permissions and run shell scripts
def set_executable_permission(script_path):
    try:
        subprocess.run(["chmod", "+x", script_path], check=True)
        print(f"Permission set for {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error setting permission: {e.stderr}")
        raise

def run_shell_script(script_path):
    try:
        result = subprocess.run([script_path], check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        raise

# Set executable permission and run the authentication script
# script_path = "./do.sh"  # Path to your shell script
# set_executable_permission(script_path)
# run_shell_script(script_path)


class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5

         # Whether or not to update/refine summaries and titles as you get new information
        self.generate_new_metadata_ind = True
        self.print_logging = True

        self.llm_url = "http://localhost:11434/api/generate"

    def llama2_generate(self, prompt):
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": "llama2",
            "prompt": prompt
        }
        try:
            response = requests.post(self.llm_url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                # Split the response text by newline to handle multiple JSON objects
                responses = response.text.split('\n')
                complete_response = ""
                for response_text in responses:
                    if response_text.strip():  # Check if the response text is not empty
                        try:
                            json_response = json.loads(response_text)
                            complete_response += json_response.get('response', '')
                            if json_response.get('done', False):
                                break
                        except json.JSONDecodeError as e:
                            return (f"JSON decode error: {e}")
                        
                return complete_response
            else:
                raise Exception(f"Failed to generate text: {response.text}")
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

    def add_propositions(self, propositions, document_id):
        for proposition in propositions:
            self.add_proposition(proposition, proposition, document_id)
    
    def add_proposition(self, proposition, content, document_id):
        if self.print_logging:
            print(f"\nAdding: '{proposition}'")

        # If it's your first chunk, just make a new chunk and don't check for others
        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition, content, document_id)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id and chunk_id.strip() in self.chunks:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id.strip()]['chunk_id']}), adding to: {self.chunks[chunk_id.strip()]['title']}")
            self.add_proposition_to_chunk(chunk_id.strip(), proposition, content)
            return
        else:
            if self.print_logging:
                print("No chunks found")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition, content, document_id)
        
    def add_proposition_to_chunk(self, chunk_id, proposition, content):
        # Add then
        self.chunks[chunk_id]['propositions'].append(proposition)
        self.chunks[chunk_id]['contents'].append(content)

        # Then grab a new summary
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunk's current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk's new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        prompt = PROMPT.format(
            proposition="\n".join(chunk['propositions']),
            current_summary=chunk['summary']
        )

        new_chunk_summary = self.llama2_generate(prompt)
        time.sleep(10)  # Add delay

        return new_chunk_summary
    
    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary, and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        prompt = PROMPT.format(
            proposition="\n".join(chunk['propositions']),
            current_summary=chunk['summary'],
            current_title=chunk['title']
        )

        updated_chunk_title = self.llama2_generate(prompt)
        time.sleep(10)  # Add delay

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        prompt = PROMPT.format(proposition=proposition)

        new_chunk_summary = self.llama2_generate(prompt)
        time.sleep(10)  # Add delay

        return new_chunk_summary
    
    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about.

                    You will be given a summary of a chunk which needs a title.

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        prompt = PROMPT.format(summary=summary)

        new_chunk_title = self.llama2_generate(prompt)
        time.sleep(10)  # Add delay

        return new_chunk_title

    def _create_new_chunk(self, proposition, content, document_id):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'document_id': document_id,
            'summary': new_chunk_summary,
            'title': new_chunk_title,
            'propositions': [proposition],
            'contents': [content]
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}), title: {new_chunk_title}")
    
    def _find_relevant_chunk(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                   "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.

                    You should determine the most relevant chunk for the following proposition.

                    You will be given a list of chunk summaries and titles, as well as the new proposition.

                    You should respond with the ID of the most relevant chunk. If no chunk is relevant, just respond with "None".

                    Example:
                    Chunk Summaries and Titles:
                    - Chunk 1 (Date & Times): This chunk contains information about dates and times that the author talks about
                    - Chunk 2 (Food): This chunk contains information about the types of food the author likes to eat

                    Proposition: Greg likes to eat sushi
                    
                    Response: Chunk 2
                    
                    Example:
                    Chunk Summaries and Titles:
                    - Chunk 1 (Date & Times): This chunk contains information about dates and times that the author talks about
                    - Chunk 2 (Food): This chunk contains information about the types of food the author likes to eat

                    Proposition: Greg likes to play soccer
                    
                    Response: None

                    """,
                ),
                ("user", "Chunk Summaries and Titles:\n{chunk_summaries}\n\nProposition:\n{proposition}"),
            ]
        )

        chunk_summaries = "\n".join([f"{key}: {value['summary']}" for key, value in self.chunks.items()])

        prompt = PROMPT.format(proposition=proposition, chunk_summaries=chunk_summaries)

        most_relevant_chunk = self.llama2_generate(prompt)
        time.sleep(10)  # Add delay

        return most_relevant_chunk

def extract_text_from_all_pages(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=20,
    length_function=len,
)

# Function to process a PDF document
def process_document(pdf_file_path):
    agentic_chunker = AgenticChunker()

    with open(pdf_file_path, 'rb') as file:
        pdf_file = file.read()

    document_id = str(uuid.uuid4())
    extracted_text = extract_text_from_all_pages(pdf_file)
    chunks = text_splitter.split_text(extracted_text)

    for chunk in chunks:
        agentic_chunker.add_proposition(chunk, chunk, document_id)

    structured_chunks = agentic_chunker.chunks

    # Create Document objects with original content, summary, and title
    documents = [
        Document(
            page_content=" ".join(chunk['contents']),
            metadata={"title": chunk['title'], "summary": chunk['summary'], "document_id": chunk['document_id']}
        )
        for chunk in structured_chunks.values()
    ]

    # Initialize the text embedding model
    embedding_model = Text2vecEmbeddings()

    # Generate embeddings for the structured chunks with a 10-second cooldown
    document_embeddings = []
    for doc in documents:
        embedding = embedding_model.embed_query([doc.page_content]).tolist()[0]
        document_embeddings.append((doc, embedding))
        time.sleep(10)  # Cooldown period of 10 seconds

    # Prepare documents with embeddings for MongoDB
    mongodb_documents = [
        {
            "content": doc.page_content,
            "title": doc.metadata["title"],
            "summary": doc.metadata["summary"],
            "document_id": doc.metadata["document_id"],
            "embedding": embedding
        }
        for doc, embedding in document_embeddings
    ]

    # Insert documents into MongoDB
    collection.insert_many(mongodb_documents)

    print(f"Documents and embeddings saved to MongoDB for document {pdf_file_path}.")

# Function to monitor and process documents from a folder
def monitor_and_process_documents(input_folder_path, processing_folder_path, processed_folder_path):
    while True:
        pdf_file_paths = [os.path.join(input_folder_path, file) for file in os.listdir(input_folder_path) if file.lower().endswith('.pdf')]

        if pdf_file_paths:
            for pdf_file_path in pdf_file_paths:
                print(f"Found new document: {pdf_file_path}")

                # Move the file to the processing folder
                processing_file_path = os.path.join(processing_folder_path, os.path.basename(pdf_file_path))
                print(f"Moving document to processing folder: {processing_file_path}")
                shutil.move(pdf_file_path, processing_file_path)

                # Process the document
                print(f"Processing document: {processing_file_path}")
                process_document(processing_file_path)

                # Move the processed file to the processed folder
                processed_file_path = os.path.join(processed_folder_path, os.path.basename(processing_file_path))
                print(f"Moving processed document to processed folder: {processed_file_path}")
                shutil.move(processing_file_path, processed_file_path)

                print(f"Document processing completed: {pdf_file_path}")
        else:
            print("Waiting for new documents...")

        # Wait for a while before checking again
        time.sleep(10)

# Folder paths
input_folder_path = 'input_pdfs'  # Folder where new PDFs will be added
processing_folder_path = 'processing_pdfs'  # Folder where PDFs will be moved for processing
processed_folder_path = 'processed_pdfs'  # Folder where processed PDFs will be moved

# Ensure the folders exist
os.makedirs(input_folder_path, exist_ok=True)
os.makedirs(processing_folder_path, exist_ok=True)
os.makedirs(processed_folder_path, exist_ok=True)

# Start monitoring and processing documents
monitor_and_process_documents(input_folder_path, processing_folder_path, processed_folder_path)
