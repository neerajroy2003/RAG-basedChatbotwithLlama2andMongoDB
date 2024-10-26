import asyncio
import logging
import uuid
import json
import requests
from pymongo import MongoClient, ASCENDING
import bcrypt
import gradio as gr
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.embeddings.text2vec import Text2vecEmbeddings

# MongoDB connection URI
MONGODB_URI = "mongodb+srv://markorganisation:bS7HUOnSsADbp8lY@chat.xgrad5y.mongodb.net/?retryWrites=true&w=majority&appName=chat"

client = MongoClient(MONGODB_URI)
# Database and collection names
DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
HISTORY_COLLECTION_NAME = "history"
USERS_COLLECTION_NAME = "users"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

class ListConvertedText2vecEmbeddings(Text2vecEmbeddings):
    def embed_documents(self, texts):
        return [super().embed_documents(text).tolist() for text in texts]
    
    def embed_query(self, text):
        return super().embed_query(text).tolist()

embeddings = ListConvertedText2vecEmbeddings()

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="content"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def format_context(docs):
    context_parts = []
    for doc in docs:
        context_parts.append(f"Title: {doc.metadata['title']}\nSummary: {doc.metadata['summary']}\nContent: {doc.page_content}")
    return "\n\n".join(context_parts)

retrieve = {"context": retriever | format_context, "question": RunnablePassthrough()}

template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Define classes to simulate ChatPromptValue, SystemMessage, and HumanMessage
class ChatPromptValue:
    def __init__(self, messages):
        self.messages = messages

class SystemMessage:
    def __init__(self, content):
        self.role = "system"
        self.content = content

class HumanMessage:
    def __init__(self, content):
        self.role = "user"
        self.content = content

def format_prompt(chat_prompt_value):
    
    formatted_messages = []
    for i, message in enumerate(chat_prompt_value.messages):
        role = 'System' if i % 2 == 0 else 'User'
        formatted_messages.append({"role": role, "content": message.content})
        
    formatted_prompt = {
        "model": "llama2",
        "prompt": str({
            "messages": formatted_messages
        })
    }
    return formatted_prompt

class Llama2Model:
    def __init__(self, llm_url):
        self.llm_url = llm_url

    def generate(self, chat_prompt_value):
        prompt = format_prompt(chat_prompt_value)
        headers = {'Content-Type': 'application/json'}
        try:
            data=json.dumps(prompt)
            print(data)
            response = requests.post(self.llm_url, headers=headers, data=data)
            
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
                            return f"JSON decode error: {e}"
                return complete_response
            else:
                raise Exception(f"Failed to generate text: {response.text}")
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

llm_url = "http://localhost:11434/api/generate"
model = Llama2Model(llm_url)

parse_output = StrOutputParser()

naive_rag_chain = (
    retrieve
    | prompt
    | model.generate
    | parse_output
)

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(MONGODB_URI, session_id, database_name=DB_NAME, collection_name=HISTORY_COLLECTION_NAME)

async def remove_oldest_conversation_if_needed(session_id: str):
    try:
        history_collection = client[DB_NAME][HISTORY_COLLECTION_NAME]
        message_count = history_collection.count_documents({"SessionId": session_id})
        print(f"Total messages for session {session_id}: {message_count}")
        if message_count >= 8:
            oldest_messages = list(history_collection.find({"SessionId": session_id}).sort("_id", ASCENDING).limit(2))
            oldest_ids = [msg["_id"] for msg in oldest_messages]
            print(f"Messages to remove: {oldest_ids}")
            if oldest_ids:
                result = history_collection.delete_many({"_id": {"$in": oldest_ids}})
                print(f"Removed {result.deleted_count} oldest messages for session {session_id}.")
            else:
                print("No messages found to remove.")
    except Exception as e:
        print(f"Error in clearing history for session {session_id}: {e}")

async def clear_orphaned_history_messages():
    try:
        history_collection = client[DB_NAME][HISTORY_COLLECTION_NAME]
        users_collection = client[DB_NAME][USERS_COLLECTION_NAME]
        user_session_ids = users_collection.distinct("session_id")
        print(f"User session IDs to keep: {user_session_ids}")
        history_session_ids = history_collection.distinct("SessionId")
        print(f"History session IDs found: {history_session_ids}")
        orphaned_session_ids = set(history_session_ids) - set(user_session_ids)
        print(f"Orphaned session IDs to remove: {orphaned_session_ids}")
        if orphaned_session_ids:
            result = history_collection.delete_many({
                "SessionId": {"$in": list(orphaned_session_ids)}
            })
            print(f"Removed {result.deleted_count} orphaned messages from history.")
    except Exception as e:
        print(f"Error in clearing orphaned history messages: {e}")

async def clear_history(session_id: str):
    try:
        history_collection = client[DB_NAME][HISTORY_COLLECTION_NAME]
        message_count = history_collection.count_documents({"SessionId": session_id})
        print(f"Total messages for session {session_id}: {message_count}")
        if message_count > 0:
            result = history_collection.delete_many({"SessionId": session_id})
            print(f"Removed {result.deleted_count} messages for session {session_id}.")
        return {'status': 'success', 'message': 'All chat history cleared'}
    except Exception as e:
        print(f"Error in clearing history for session {session_id}: {e}")
        return {'status': 'error', 'message': str(e)}

standalone_system_prompt = """
Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
Only return the final standalone question. \
"""
standalone_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", standalone_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

question_chain = standalone_question_prompt | model.generate | parse_output
retriever_chain = RunnablePassthrough.assign(context=question_chain | retriever | format_context)

rag_system_prompt = """Answer the question based only on the following context: \
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

rag_chain = (
    retriever_chain
    | rag_prompt
    | model.generate
    | parse_output
)

with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

async def register_user(username: str, password: str):
    users_collection = client[DB_NAME][USERS_COLLECTION_NAME]
    hashed_password = hash_password(password)
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "session_id": None
    })
    return {'status': 'success', 'message': f"user {username} registered successfully."}

async def login_user(username: str, password: str) -> dict:
    users_collection = client[DB_NAME][USERS_COLLECTION_NAME]
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user['password']):
        session_id = str(uuid.uuid4())
        users_collection.update_one({"username": username}, {"$set": {"session_id": session_id}})
        return {'status': 'success', 'session_id': session_id}
    else:
        return {'status': 'error', 'message': 'Invalid user code or password'}

async def fetch_user_id(username: str, password: str) -> str:
    users_collection = client[DB_NAME][USERS_COLLECTION_NAME]
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user['password']):
        return user.get("session_id")
    return None

async def main_chatbot(username: str, password: str, user_input: str) -> dict:
    await clear_orphaned_history_messages()
    user_id = await fetch_user_id(username, password)
    if not user_id:
        return {'status': 'error', 'message': 'Invalid user code or password'}
    await remove_oldest_conversation_if_needed(user_id)
    response = with_message_history.invoke({"question": user_input}, {"configurable": {"session_id": user_id}})
    return {'status': 'success', 'answer': response}

async def register_user_interface(username: str, password: str):
    result = await register_user(username, password)
    return result['message']

async def login_user_interface(username: str, password: str):
    result = await login_user(username, password)
    if result['status'] == 'success':
        user_credentials = {'username': username, 'password': password}
        return f"Login successful! Session ID: {result['session_id']}", user_credentials
    else:
        return result['message'], None

async def chat_interface(user_input: str,h, user_credentials: dict):
    if not user_credentials:
        return "You need to log in first."
    username = user_credentials['username']
    password = user_credentials['password']
    result = await main_chatbot(username, password, user_input)
    if result['status'] == 'success':
        return result['answer']
    else:
        return result['message']

async def clear_chat_history_interface(user_credentials: dict):
    if not user_credentials:
        return "You need to log in first."
    username = user_credentials['username']
    password = user_credentials['password']
    user_id = await fetch_user_id(username, password)
    if user_id:
        result = await clear_history(user_id)
        return result['message']
    else:
        return "Invalid user code or password"


custom_css = """
.chat-container {
height: 80vh;
}"""
with gr.Blocks(css=custom_css,theme=gr.themes.Monochrome()) as demo:
    user_credentials = gr.State(None)

    with gr.Tab("Register"):
        with gr.Row():
            username_reg = gr.Textbox(label="UserName")
            password_reg = gr.Textbox(label="Password", type="password")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Output")
        
        register_btn.click(register_user_interface, inputs=[username_reg, password_reg], outputs=register_output)

    with gr.Tab("Login"):
        with gr.Row():
            username_login = gr.Textbox(label="user Code")
            password_login = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Output")
        
        login_btn.click(
            login_user_interface, 
            inputs=[username_login, password_login], 
            outputs=[login_output, user_credentials]
        )
        
    with gr.Tab("Chat"):
        with gr.Column(elem_classes="chat-container") as chat_container:
            gr.ChatInterface(fn=chat_interface, additional_inputs=[user_credentials], title="Chat Bot")
    
    with gr.Tab("Clear Chat History"):
        clear_history_btn = gr.Button("Clear History")
        clear_history_output = gr.Textbox(label="Output")

        clear_history_btn.click(clear_chat_history_interface, inputs=[user_credentials], outputs=clear_history_output)

# Launch the Gradio interface
demo.launch()
