import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

class RAGSystem:
    def __init__(self):
        # Inizializza ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Inizializza il modello di embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Crea o recupera la collection
        try:
            self.collection = self.client.get_collection(name="file_metadata")
        except:
            self.collection = self.client.create_collection(
                name="file_metadata",
                metadata={"hnsw:space": "cosine"}
            )
    
    async def index_files(self, file_data: List[Dict[str, Any]]):
        """Indicizza i dati dei file nel vector database"""
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        for i, file_info in enumerate(file_data):
            # Crea un documento testuale per l'embedding
            doc_text = self._create_document_text(file_info)
            documents.append(doc_text)
            
            # Genera l'embedding per il documento
            embedding = self.embedding_model.encode(doc_text).tolist()
            embeddings.append(embedding)
            
            # Prepara i metadati (ChromaDB supporta solo valori semplici)
            metadata = {
                "path": file_info["path"],
                "name": file_info["name"],
                "type": file_info["type"],
                "size": file_info.get("size", 0),
                "size_human": file_info.get("size_human", ""),
                "extension": file_info.get("extension", ""),
                "mime_type": file_info.get("mime_type", ""),
                "created": file_info.get("created", ""),
                "modified": file_info.get("modified", ""),
                "cleanup_suggestions": json.dumps(file_info.get("cleanup_suggestions", []))
            }
            
            metadatas.append(metadata)
            # Crea un ID unico basato sul percorso del file
            unique_id = f"file_{abs(hash(file_info['path']))}"
            ids.append(unique_id)
        
        # Aggiungi alla collection con embeddings, evitando duplicati
        if documents:
            try:
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                # Se ci sono ID duplicati, li aggiungiamo uno alla volta
                for doc, emb, meta, id_val in zip(documents, embeddings, metadatas, ids):
                    try:
                        self.collection.add(
                            documents=[doc],
                            embeddings=[emb],
                            metadatas=[meta],
                            ids=[id_val]
                        )
                    except:
                        # Se l'ID esiste già, aggiorniamo il documento
                        try:
                            self.collection.update(
                                documents=[doc],
                                embeddings=[emb],
                                metadatas=[meta],
                                ids=[id_val]
                            )
                        except:
                            pass  # Skip se non può aggiornare
    
    def _create_document_text(self, file_info: Dict[str, Any]) -> str:
        """Crea un testo descrittivo del file per l'embedding"""
        parts = []
        
        # Nome del file e percorso completo per migliorare la ricerca
        name = file_info['name']
        path = file_info['path']
        
        # Aggiungi il nome del file con variazioni
        parts.append(f"Nome file: {name}")
        
        # Aggiungi il percorso e le sue componenti
        path_components = path.split('/')
        if len(path_components) > 1:
            parts.append(f"Directory: {'/'.join(path_components[:-1])}")
        parts.append(f"Percorso completo: {path}")
        
        # Tipo di file e categoria
        file_type = file_info['type']
        parts.append(f"Tipo: {file_type}")
        
        # Estensione e tipo MIME per categorizzazione
        if file_info.get("extension"):
            ext = file_info['extension'].lower()
            parts.append(f"Estensione: {ext}")
            
            # Aggiungi categorie basate sull'estensione
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp']:
                parts.append("Categoria: immagine foto grafica")
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                parts.append("Categoria: video filmato")
            elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                parts.append("Categoria: audio musica suono")
            elif ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
                parts.append("Categoria: documento testo")
            elif ext in ['.zip', '.rar', '.tar', '.gz', '.7z']:
                parts.append("Categoria: archivio compresso")
            elif ext in ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.ts']:
                parts.append("Categoria: codice programmazione sorgente")
            elif ext in ['.xls', '.xlsx', '.csv']:
                parts.append("Categoria: foglio di calcolo dati tabella")
        
        # Dimensione con descrizione
        if file_info.get("size"):
            size = file_info.get("size", 0)
            size_human = file_info.get('size_human', '')
            parts.append(f"Dimensione: {size_human}")
            
            # Aggiungi descrittori di dimensione
            if size > 1024 * 1024 * 1024:  # > 1GB
                parts.append("File molto grande gigabyte")
            elif size > 100 * 1024 * 1024:  # > 100MB
                parts.append("File grande pesante")
            elif size > 10 * 1024 * 1024:  # > 10MB
                parts.append("File medio")
            elif size < 1024:  # < 1KB
                parts.append("File molto piccolo leggero")
        
        if file_info.get("mime_type"):
            parts.append(f"MIME: {file_info['mime_type']}")
        
        # Suggerimenti di pulizia
        if file_info.get("cleanup_suggestions"):
            parts.append("Suggerimenti: " + ", ".join(file_info["cleanup_suggestions"]))
        
        # Informazioni specifiche
        if file_info.get("image_info"):
            img_info = file_info["image_info"]
            parts.append(f"Immagine {img_info['width']}x{img_info['height']} in formato {img_info['format']}")
        
        if file_info.get("text_info"):
            txt_info = file_info["text_info"]
            parts.append(f"File di testo con {txt_info['lines']} righe e {txt_info['words']} parole")
        
        if file_info["type"] == "directory":
            parts.append(f"Directory con {file_info.get('file_count', 0)} file e {file_info.get('dir_count', 0)} sottodirectory")
        
        return " | ".join(parts)
    
    async def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None, n_results: int = 10) -> Dict[str, Any]:
        """Esegue una query sul sistema RAG"""
        
        # Genera l'embedding per la query
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Prepara i filtri per ChromaDB
        where_filter = {}
        if filters:
            if filters.get("min_size"):
                where_filter["size"] = {"$gte": filters["min_size"]}
            if filters.get("file_type"):
                where_filter["type"] = {"$eq": filters["file_type"]}
            if filters.get("extension"):
                where_filter["extension"] = {"$eq": filters["extension"]}
        
        # Esegue la query con embeddings
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        # Formatta i risultati
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                # Riconverte i suggerimenti da JSON
                try:
                    cleanup_suggestions = json.loads(metadata.get("cleanup_suggestions", "[]"))
                except:
                    cleanup_suggestions = []
                
                formatted_results.append({
                    "document": doc,
                    "metadata": {
                        **metadata,
                        "cleanup_suggestions": cleanup_suggestions
                    },
                    "similarity_score": 1 - distance,  # Converte distanza in similarità
                    "rank": i + 1
                })
        
        # Genera una risposta intelligente
        response_text = self._generate_response(query_text, formatted_results)
        
        return {
            "query": query_text,
            "response": response_text,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    
    def _generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Genera una risposta intelligente basata sui risultati e la query"""
        if not results:
            return "Non ho trovato file che corrispondono alla tua richiesta. Prova a essere più specifico o usa parole chiave diverse."
        
        query_lower = query.lower()
        
        # Analizza l'intento della query
        if any(word in query_lower for word in ['grande', 'grandi', 'pesante', 'pesanti', 'dimensione', 'spazio', 'mb', 'gb']):
            # Query sulla dimensione
            large_files = [r for r in results if r["metadata"].get("size", 0) > 100 * 1024 * 1024]
            total_size = sum(r["metadata"].get("size", 0) for r in results)
            
            from file_analyzer import FileAnalyzer
            analyzer = FileAnalyzer()
            
            response_parts = []
            if large_files:
                response_parts.append(f"Ho trovato {len(large_files)} file grandi che occupano molto spazio:")
                for file in sorted(large_files, key=lambda x: x["metadata"].get("size", 0), reverse=True)[:5]:
                    size_human = analyzer._format_size(file["metadata"].get("size", 0))
                    response_parts.append(f"• {file['metadata']['name']}: {size_human}")
            
            if total_size > 0:
                size_human = analyzer._format_size(total_size)
                response_parts.append(f"\nDimensione totale dei file trovati: {size_human}")
            
            return "\n".join(response_parts) if response_parts else f"Ho trovato {len(results)} file correlati alla tua ricerca."
            
        elif any(word in query_lower for word in ['immagine', 'immagini', 'foto', 'video', 'audio', 'musica']):
            # Query per tipo di media
            media_files = [r for r in results if r["metadata"].get("extension", "").lower() in 
                          ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.avi', '.mov', '.mp3', '.wav']]
            
            if media_files:
                response_parts = [f"Ho trovato {len(media_files)} file multimediali:"]
                for file in media_files[:10]:
                    response_parts.append(f"• {file['metadata']['name']} ({file['metadata'].get('size_human', 'N/A')})")
                return "\n".join(response_parts)
            
        elif any(word in query_lower for word in ['vecchio', 'vecchi', 'obsoleto', 'inutile', 'eliminare', 'cancellare', 'pulire']):
            # Query per pulizia
            response_parts = [f"Ho trovato {len(results)} file che potrebbero essere candidati per la pulizia:"]
            
            # Raggruppa per suggerimenti
            all_suggestions = []
            for result in results:
                suggestions = result["metadata"].get("cleanup_suggestions", [])
                if suggestions:
                    all_suggestions.extend(suggestions)
                    response_parts.append(f"• {result['metadata']['name']}: {', '.join(suggestions)}")
            
            if all_suggestions:
                suggestion_counts = {}
                for suggestion in all_suggestions:
                    suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
                
                most_common = max(suggestion_counts.items(), key=lambda x: x[1])
                response_parts.append(f"\nSuggerimento principale: {most_common[0]} ({most_common[1]} file)")
            
            return "\n".join(response_parts)
        
        # Risposta generica migliorata
        response_parts = [f"Ho trovato {len(results)} file rilevanti per '{query}':"]
        
        # Mostra i primi 5 risultati più rilevanti
        for result in results[:5]:
            similarity = result.get('similarity_score', 0)
            response_parts.append(f"• {result['metadata']['name']} ({result['metadata'].get('size_human', 'N/A')}) - {int(similarity * 100)}% rilevante")
        
        # Analizza i pattern nei risultati
        total_size = sum(r["metadata"].get("size", 0) for r in results)
        if total_size > 0:
            from file_analyzer import FileAnalyzer
            analyzer = FileAnalyzer()
            size_human = analyzer._format_size(total_size)
            response_parts.append(f"\nDimensione totale: {size_human}")
        
        return "\n".join(response_parts)
    
    async def get_file_type_distribution(self) -> Dict[str, Any]:
        """Restituisce la distribuzione dei tipi di file nell'indice"""
        try:
            # Recupera tutti i documenti
            all_results = self.collection.get()
            
            type_counts = {}
            size_by_type = {}
            
            for metadata in all_results["metadatas"]:
                file_type = metadata.get("type", "unknown")
                size = metadata.get("size", 0)
                
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
                size_by_type[file_type] = size_by_type.get(file_type, 0) + size
            
            return {
                "type_counts": type_counts,
                "size_by_type": size_by_type,
                "total_files": len(all_results["metadatas"])
            }
        except:
            return {"type_counts": {}, "size_by_type": {}, "total_files": 0}
    
    async def get_vectors_for_visualization(self):
        """Ottiene i vettori e metadati per la visualizzazione grafica"""
        try:
            # Ottieni tutti i documenti dalla collection
            results = self.collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not results["ids"]:
                return {"vectors": [], "metadata": [], "total": 0}
            
            # Riduci la dimensionalità per la visualizzazione (da 384 a 2D)
            # Useremo PCA per la riduzione
            from sklearn.decomposition import PCA
            import numpy as np
            
            embeddings = np.array(results["embeddings"])
            
            # Se abbiamo abbastanza documenti, usa PCA per ridurre a 2D
            if len(embeddings) > 1:
                pca = PCA(n_components=2)
                coords_2d = pca.fit_transform(embeddings)
            else:
                # Se c'è solo un documento, mettilo al centro
                coords_2d = [[0, 0]] * len(embeddings)
            
            # Prepara i dati per la visualizzazione
            vectors_data = []
            for i, (id, metadata, doc, coords) in enumerate(zip(
                results["ids"], 
                results["metadatas"], 
                results["documents"],
                coords_2d
            )):
                # Estrai il tipo di file per il colore
                file_type = metadata.get("type", "unknown")
                extension = metadata.get("extension", "")
                
                # Determina la categoria per il colore
                category = "other"
                if extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    category = "image"
                elif extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    category = "video"
                elif extension in ['.mp3', '.wav', '.flac']:
                    category = "audio"
                elif extension in ['.pdf', '.doc', '.docx', '.txt', '.md']:
                    category = "document"
                elif extension in ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css']:
                    category = "code"
                elif extension in ['.zip', '.rar', '.tar', '.gz']:
                    category = "archive"
                elif file_type == "directory":
                    category = "directory"
                
                vectors_data.append({
                    "id": id,
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "name": metadata.get("name", "Unknown"),
                    "path": metadata.get("path", ""),
                    "size": metadata.get("size", 0),
                    "size_human": metadata.get("size_human", ""),
                    "type": file_type,
                    "category": category,
                    "description": doc[:200] if doc else ""  # Prima parte del documento
                })
            
            return {
                "vectors": vectors_data,
                "total": len(vectors_data),
                "categories": ["image", "video", "audio", "document", "code", "archive", "directory", "other"]
            }
            
        except Exception as e:
            return {"error": str(e), "vectors": [], "total": 0}
    
    async def clear_index(self):
        """Pulisce l'indice"""
        try:
            self.client.delete_collection(name="file_metadata")
            self.collection = self.client.create_collection(
                name="file_metadata",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            pass