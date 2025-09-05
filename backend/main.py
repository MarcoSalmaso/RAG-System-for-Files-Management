from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path

from file_analyzer import FileAnalyzer
from rag_system import RAGSystem

app = FastAPI(title="File Analyzer RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_analyzer = FileAnalyzer()
rag_system = RAGSystem()

def convert_path_for_display(path: str) -> str:
    """Converte i percorsi Docker in percorsi user-friendly per la visualizzazione"""
    if path.startswith('/host_users'):
        # Ottiene il percorso home dell'utente dal sistema
        home_path = os.environ.get('HOME', '/home/user')
        return path.replace('/host_users', home_path)
    return path

def convert_results_paths(results: List[Dict]) -> List[Dict]:
    """Converte tutti i percorsi nei risultati per una visualizzazione user-friendly"""
    for item in results:
        if 'path' in item:
            item['path'] = convert_path_for_display(item['path'])
        if 'metadata' in item and 'path' in item['metadata']:
            item['metadata']['path'] = convert_path_for_display(item['metadata']['path'])
    return results

class ScanRequest(BaseModel):
    path: str
    include_hidden: bool = False
    max_depth: int = 5

class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {"message": "File Analyzer RAG System API"}

@app.post("/scan")
async def scan_directory(request: ScanRequest):
    try:
        if not os.path.exists(request.path):
            raise HTTPException(status_code=404, detail="Path not found")
        
        results = await file_analyzer.scan_directory(
            request.path, 
            request.include_hidden, 
            request.max_depth
        )
        
        # Store results in vector database
        await rag_system.index_files(results)
        
        # Converti i percorsi per la visualizzazione
        display_results = convert_results_paths(results.copy())
        
        return {
            "status": "success",
            "scanned_files": len(display_results),
            "results": display_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_files(request: QueryRequest):
    try:
        response = await rag_system.query(request.query, request.filters)
        
        # Converti i percorsi nei risultati
        if 'results' in response:
            response['results'] = convert_results_paths(response['results'].copy())
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        stats = await file_analyzer.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file-types")
async def get_file_types():
    try:
        file_types = await rag_system.get_file_type_distribution()
        return file_types
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vectors")
async def get_vectors():
    """Ottiene i vettori e metadati per la visualizzazione"""
    try:
        vectors_data = await rag_system.get_vectors_for_visualization()
        return vectors_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_index():
    try:
        await rag_system.clear_index()
        return {"status": "success", "message": "Index cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)