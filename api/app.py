# Import required FastAPI components for building the API
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic for data validation and settings management
from pydantic import BaseModel
# Import OpenAI client for interacting with OpenAI's API
from openai import OpenAI
import os
import sys
from typing import Optional

# Add the current directory to Python path for Vercel compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import RAG service with error handling for deployment
try:
    from rag_service import rag_service
except ImportError:
    # Fallback for different import paths
    try:
        import rag_service as rag_module
        rag_service = rag_module.rag_service
    except ImportError:
        rag_service = None
        print("Warning: RAG service could not be imported")

# Initialize FastAPI application with a title
app = FastAPI(title="OpenAI Chat API with PDF RAG")

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows the API to be accessed from different domains/origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,  # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers in requests
)

# Define the data model for chat requests using Pydantic
# This ensures incoming request data is properly validated
class ChatRequest(BaseModel):
    system_message: str  # Message from the developer/system
    user_message: str      # Message from the user
    model: Optional[str] = "gpt-4.1-mini"  # Optional model selection with default
    api_key: str          # OpenAI API key for authentication
    temperature: Optional[float] = 0.7  # Temperature for controlling creativity (0-2)

# Define the data model for RAG chat requests
class RAGChatRequest(BaseModel):
    user_message: str      # Message from the user
    system_message: Optional[str] = ""  # Optional system message
    api_key: str          # OpenAI API key for authentication

# Define the main chat endpoint that handles POST requests
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize OpenAI client with the provided API key
        client = OpenAI(api_key=request.api_key)
        
        # Create an async generator function for streaming responses
        async def generate():
            # Create a streaming chat completion request
            stream = client.chat.completions.create(
                model=request.model or "gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "system", "content": "Do not produce answers greater than 500 words"},
                    {"role": "system", "content": "Use $...$ for inline math and $$...$$ for block math. Do not use square brackets [ ... ] for mathematical expressions. Remove any spaces between the closing $ and the content of the expression."},
                    {"role": "user", "content": request.user_message}
                ],
                stream=True,  # Enable streaming response
                temperature=request.temperature
            )

            # Yield each chunk of the response as it becomes available
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        # Handle any errors that occur during processing
        raise HTTPException(status_code=500, detail=str(e))

# Define PDF upload and indexing endpoint
@app.post("/api/upload-data-file")
async def upload_data_file(file: UploadFile = File(...), api_key: str = Form(...)):
    """Upload and index a PDF file for RAG functionality."""
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service is not available")
    
    try:
        result = await rag_service.upload_and_index_data_source(file, api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define RAG chat endpoint
@app.post("/api/rag-chat")
async def rag_chat(request: RAGChatRequest):
    """Chat with the indexed PDF or TXT input using RAG with streaming."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service is not available")
    
    try:
        # Create an async generator function for streaming responses
        async def generate():
            # Type checking ensures rag_service is not None here
            if rag_service is not None:
                async for chunk in rag_service.chat_with_data_sources_stream(
                    request.user_message, 
                    request.api_key, 
                    request.system_message or ""
                ):
                    yield chunk

        # Return a streaming response to the client
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define endpoint to get PDF indexing status
@app.get("/api/data-file-indexing-status")
async def get_data_file_indexing_status():
    """Get the current status of Data File indexing."""
    if rag_service is None:
        return {"is_indexed": False, "document_id": None, "chunks_count": 0, "error": "RAG service not available"}
    
    return rag_service.get_index_status()

# Define endpoint to clear the indexed data
@app.delete("/api/clear-data-file-index")
async def clear_data_file_index():
    """Clear the currently indexed data file."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service is not available")
    
    return rag_service.clear_index()

# Define endpoint to get file history
@app.get("/api/file-history")
async def get_file_history():
    """Get the history of all uploaded files."""
    if rag_service is None:
        return {"success": False, "file_history": [], "total_files": 0, "error": "RAG service not available"}
    
    return rag_service.get_file_history()

# Define a health check endpoint to verify API status
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    # Start the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
