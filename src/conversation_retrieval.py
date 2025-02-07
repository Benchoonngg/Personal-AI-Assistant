from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import json
import os
import shutil
from datetime import datetime

class ConversationRetrieval:
    def __init__(self, history_dir="history"):
        self.history_dir = history_dir
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        os.makedirs(self.history_dir, exist_ok=True)
        
        if not os.path.exists("./chroma_db"):
            self.refresh_vector_store()
        else:
            try:
                self.vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
                print("Loaded existing vector store")
            except Exception as e:
                print(f"Error loading vector store: {e}")
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
                            conversation_text = []
                            for msg in data.get('conversation', []):
                                if msg['role'] != 'system':
                                    formatted_msg = f"[{msg['role']}]: {msg['content']}"
                                    conversation_text.append(formatted_msg)
                            if conversation_text:
                                conversations.append("\n".join(conversation_text))
                    except Exception as e:
                        print(f"Error reading file {filename}: {e}")
                        continue
            
            print(f"Loaded conversations from {len(os.listdir(self.history_dir))} files")
            return conversations
            
        except Exception as e:
            print(f"Error loading conversation files: {e}")
            return []

    def refresh_vector_store(self):
        """Create new vector store from current history"""
        try:
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                print("Cleared previous vector store")

            conversations = self.load_conversation_files()
            if not conversations:
                print("No conversations found in history")
                self.vector_store = None
                return

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

    def get_relevant_history(self, query, k=5):
        """Get relevant conversation chunks"""
        try:
            if not hasattr(self, 'vector_store') or self.vector_store is None:
                print("No vector store available")
                return []

            # Create multiple search queries for better coverage
            search_queries = [
                query,  # Original query
                " ".join([word for word in query.split() if len(word) > 3]),  # Key words only
                "personal information name job role",  # Always include personal context
            ]
            
            all_results = []
            for q in search_queries:
                results = self.vector_store.similarity_search(q, k=k)
                all_results.extend([doc.page_content for doc in results])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for result in all_results:
                if result not in seen:
                    seen.add(result)
                    unique_results.append(result)
            
            return unique_results[:k]  # Return top k unique results

        except Exception as e:
            print(f"Error retrieving history: {e}")
            return []

    def clear_history(self, delete_files=False):
        """Clear vector store and optionally delete JSON files"""
        try:
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                print("Cleared vector store")

            if delete_files:
                for file in os.listdir(self.history_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.history_dir, file))
                print("Deleted conversation files")

        except Exception as e:
            print(f"Error clearing history: {e}")
