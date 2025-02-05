from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json
import os
from datetime import datetime

class ConversationRetrieval:
    def __init__(self, history_dir="history"):
        self.history_dir = history_dir
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_conversation_files(self):
        conversations = []
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.history_dir, filename), 'r') as f:
                    data = json.load(f)
                    # Format conversations for better context
                    for msg in data['conversation']:
                        if msg['role'] != 'system':  # Skip system messages
                            formatted_msg = f"[{msg['role']}]: {msg['content']}"
                            conversations.append(formatted_msg)
        return conversations

    def create_vector_store(self):
        conversations = self.load_conversation_files()
        if not conversations:
            return None
            
        # Split conversations into chunks
        docs = self.text_splitter.create_documents(conversations)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store

    def get_relevant_history(self, query, k=3):
        try:
            vector_store = self.create_vector_store()
            if not vector_store:
                return []
                
            # Search for relevant conversation chunks
            results = vector_store.similarity_search(query, k=k)
            
            # Format results
            relevant_history = []
            for doc in results:
                relevant_history.append(doc.page_content)
                
            return relevant_history
        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []

    def clear_history(self, delete_files=False):
        try:
            # Clear Chroma database
            if os.path.exists("./chroma_db"):
                import shutil
                shutil.rmtree("./chroma_db")
                print("Cleared vector store")
                
            # Optionally delete history files
            if delete_files:
                for file in os.listdir(self.history_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.history_dir, file))
                print("Deleted conversation files")
                
        except Exception as e:
            print(f"Error clearing history: {e}")
