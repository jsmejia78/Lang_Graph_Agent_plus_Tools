import os
import tempfile
import uuid
import sys
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException

# Add parent directory to Python path to access aimakerspace at root level
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from aimakerspace.text_utils import TextFileLoader, PDFLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import asyncio


class RAGService:
    def __init__(self):
        self.vector_db = None
        self.chat_model = None
        self.embedding_model = None
        self.document_chunks = []
        self.document_id = None
        # Store detailed file information
        self.file_info = {
            'filename': None,
            'upload_timestamp': None,
            'file_type': None,
            'file_size': None
        }
        # Store file history
        self.file_history = []
        
    async def initialize_models(self, api_key: str):
        """Initialize the embedding and chat models with the provided API key."""
        try:
            # Set the API key as environment variable for the models
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Initialize models
            self.embedding_model = EmbeddingModel()
            self.chat_model = ChatOpenAI()
            
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize models: {str(e)}")
    
    async def upload_and_index_data_source(self, file: UploadFile, api_key: str) -> Dict[str, Any]:
        """Upload a PDF or TXT file and index it for RAG functionality."""
        try:
            # Initialize models if not already done
            if not self.embedding_model:
                await self.initialize_models(api_key)
            
            # Validate file type
            if not file.filename or \
                (not file.filename.lower().endswith('.txt') and \
                 not file.filename.lower().endswith('.pdf')):
                raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
            
            # Store file information
            import datetime
            self.file_info = {
                'filename': file.filename,
                'upload_timestamp': datetime.datetime.now().isoformat(),
                'file_type': 'PDF' if file.filename.lower().endswith('.pdf') else 'TXT',
                'file_size': file.size if hasattr(file, 'size') else None
            }
            
            if file.filename.lower().endswith('.pdf'):
                # Create a temporary file to store the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    # Update file size if not available
                    if not self.file_info['file_size']:
                        self.file_info['file_size'] = len(content)
            elif file.filename.lower().endswith('.txt'):
                # Create a temporary file to store the uploaded TXT
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                    # Update file size if not available
                    if not self.file_info['file_size']:
                        self.file_info['file_size'] = len(content)

            try:

                if file.filename.lower().endswith('.pdf'):
                    # Load and extract text from PDF
                    pdf_loader = PDFLoader(temp_file_path)
                    documents = pdf_loader.load_documents()

                elif file.filename.lower().endswith('.txt'):
                    # Load and extract text from TXT
                    text_loader = TextFileLoader(temp_file_path)
                    documents = text_loader.load_documents()

                if not documents or not documents[0].strip():
                    raise HTTPException(status_code=400, detail="Could not extract text from Data Source (TXT, PDF). The file might be empty, corrupted, or contain only images.")
                    
                # Split text into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                self.document_chunks = text_splitter.split_texts(documents)
                
                # if embedding model is initialized, create vector database and index the chunks
                if self.embedding_model:
                    if self.vector_db is None:
                        self.vector_db = VectorDatabase(self.embedding_model)
                    # index the chunks
                    await self.vector_db.abuild_from_list(self.document_chunks)
                
                # Generate a unique document ID
                self.document_id = str(uuid.uuid4())
                
                # Add file to history
                file_history_entry = {
                    'document_id': self.document_id,
                    'filename': self.file_info['filename'],
                    'file_type': self.file_info['file_type'],
                    'file_size': self.file_info['file_size'],
                    'upload_timestamp': self.file_info['upload_timestamp'],
                    'chunks_count': len(self.document_chunks),
                    'is_current': True
                }
                
                # Mark all previous files as not current
                for entry in self.file_history:
                    entry['is_current'] = False
                
                # Add new file to history
                self.file_history.append(file_history_entry)
                
                return {
                    "success": True,
                    "message": f"File indexed successfully. Extracted {len(self.document_chunks)} chunks from {file.filename}",
                    "document_id": self.document_id,
                    "chunks_count": len(self.document_chunks),
                    "file_info": self.file_info
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process Data Source (TXT, PDF): {str(e)}")


    
    async def chat_with_data_sources_stream(self, user_message: str, api_key: str, system_message: str = ""):
        """Chat with the indexed data source using RAG with streaming."""
        try:
            # Check if data source is indexed
            if not self.vector_db or not self.document_chunks:
                raise HTTPException(status_code=400, detail="No data source has been indexed. Please upload and index a PDF or TXT file first.")
            
            # Initialize models if not already done
            if not self.chat_model:
                await self.initialize_models(api_key)
            
            # Search for relevant chunks
            search_results = self.vector_db.search_by_text(user_message, k=3, return_as_text=True)  # type: ignore
            relevant_chunks = search_results if search_results else []
            
            # Create context from relevant chunks
            context = "\n\n".join(relevant_chunks) if relevant_chunks else ""  # type: ignore
            
            # Create the prompt with context
            if system_message:
                full_system_message = f"{system_message}\n\nUse the following context to answer the user's question:\n\n{context}"
            else:
                full_system_message = f"You are a helpful assistant. Use the following context from the uploaded document to answer the user's question. If the context doesn't contain enough information to answer the question, say so.\n\nContext:\n{context}"
            
            # Generate streaming response using chat model
            if self.chat_model:
                messages = [
                    {"role": "system", "content": full_system_message},
                    {"role": "user", "content": user_message}
                ]
                async for chunk in self.chat_model.astream(messages):
                    yield chunk
            else:
                raise HTTPException(status_code=500, detail="Chat model not initialized")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

    async def chat_with_data_sources(self, user_message: str, api_key: str, system_message: str = "") -> str:
        """Chat with the indexed data source using RAG."""
        try:
            # Check if data source is indexed
            if not self.vector_db or not self.document_chunks:
                raise HTTPException(status_code=400, detail="No data source has been indexed. Please upload and index a PDF or TXT file first.")
            
            # Initialize models if not already done
            if not self.chat_model:
                await self.initialize_models(api_key)
            
            # Search for relevant chunks
            search_results = self.vector_db.search_by_text(user_message, k=3, return_as_text=True)  # type: ignore
            relevant_chunks = search_results if search_results else []
            
            # Create context from relevant chunks
            context = "\n\n".join(relevant_chunks) if relevant_chunks else ""  # type: ignore
            
            # Create the prompt with context
            if system_message:
                full_system_message = f"{system_message}\n\nUse the following context to answer the user's question:\n\n{context}"
            else:
                full_system_message = f"You are a helpful assistant. Use the following context from the uploaded document to answer the user's question. If the context doesn't contain enough information to answer the question, say so.\n\nContext:\n{context}"
            
            # Generate response using chat model
            if self.chat_model:
                messages = [
                    {"role": "system", "content": full_system_message},
                    {"role": "user", "content": user_message}
                ]
                response = ""
                async for chunk in self.chat_model.astream(messages):
                    response += chunk
                return response
            else:
                raise HTTPException(status_code=500, detail="Chat model not initialized")
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    def get_index_status(self) -> Dict[str, Any]:
        """Get the current status of the data source indexing."""
        return {
            "is_indexed": self.vector_db is not None,
            "document_id": self.document_id,
            "chunks_count": len(self.document_chunks) if self.document_chunks else 0,
            "file_info": self.file_info if self.vector_db is not None else None
        }
    
    def get_file_history(self) -> Dict[str, Any]:
        """Get the history of all uploaded files."""
        return {
            "success": True,
            "file_history": self.file_history,
            "total_files": len(self.file_history)
        }
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear the current indexed data and reset the service."""
        self.vector_db = None
        self.document_chunks = []
        self.document_id = None
        self.file_info = {
            'filename': None,
            'upload_timestamp': None,
            'file_type': None,
            'file_size': None
        }
        # Clear the file history completely
        self.file_history = []
        # Note: We keep the models initialized to avoid re-initialization overhead
        return {
            "success": True,
            "message": "Index and file history cleared successfully"
        }


# Global RAG service instance
rag_service = RAGService() 