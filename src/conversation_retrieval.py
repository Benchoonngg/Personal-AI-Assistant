from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import json
import os
import shutil
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
        
        # Ensure history directory exists
        os.makedirs(self.history_dir, exist_ok=True)
        
        # Initialize vector store
        self.refresh_vector_store()
        print("Conversation retrieval initialized!")

    def load_conversation_files(self):
        """Load all conversations from JSON files"""
        conversations = []
        try:
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.history_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for msg in data.get('conversation', []):
                                if msg['role'] != 'system':
                                    formatted_msg = f"[{msg['role']}]: {msg['content']}"
                                    conversations.append(formatted_msg)
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
                        continue
            
            print(f"Loaded {len(conversations)} messages from {len(os.listdir(self.history_dir))} files")
            return conversations
            
        except Exception as e:
            print(f"Error loading conversation files: {e}")
            return []

    def refresh_vector_store(self):
        """Create new vector store from current history"""
        try:
            # Remove old vector store if exists
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                print("Cleared previous vector store")

            # Load conversations
            conversations = self.load_conversation_files()
            if not conversations:
                print("No conversations found in history")
                self.vector_store = None
                return

            # Create new vector store
            print("Creating new vector store...")
            docs = self.text_splitter.create_documents(conversations)
            self.vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            print(f"Vector store created with {len(docs)} chunks")

        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vector_store = None

    def get_relevant_history(self, query, k=3):
        """Get relevant conversation chunks"""
        try:
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                print("No vector store available")
                return []

            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]

        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []

    def clear_history(self, delete_files=False):
        """Clear vector store and optionally delete JSON files"""
        try:
            # Always clear vector store
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                print("Cleared vector store")

            # Optionally delete JSON files
            if delete_files:
                for file in os.listdir(self.history_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.history_dir, file))
                print("Deleted conversation files")

        except Exception as e:
            print(f"Error clearing history: {e}")
