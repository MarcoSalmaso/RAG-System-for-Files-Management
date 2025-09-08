import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import httpx
import asyncio

class RAGSystem:
    def __init__(self):
        # Inizializza ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Inizializza il modello di embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configurazione LM Studio API  
        self.lm_studio_url = "http://host.docker.internal:1234/v1/chat/completions"
        self.lm_studio_available = False
        self._check_lm_studio_availability()
        
        # Track the most recently scanned path for context
        self.last_scanned_path = None
        
        # Crea o recupera la collection
        try:
            self.collection = self.client.get_collection(name="file_metadata")
        except:
            self.collection = self.client.create_collection(
                name="file_metadata",
                metadata={"hnsw:space": "cosine"}
            )
    
    def _check_lm_studio_availability(self):
        """Controlla se LM Studio è disponibile in modo asincrono"""
        import threading
        
        def check_availability():
            try:
                import requests
                response = requests.get("http://host.docker.internal:1234/v1/models", timeout=2)
                if response.status_code == 200:
                    self.lm_studio_available = True
                    print("✅ LM Studio rilevato e disponibile!")
                else:
                    self.lm_studio_available = False
                    print("⚠️ LM Studio non risponde correttamente")
            except Exception:
                self.lm_studio_available = False
                print("⚠️ LM Studio non disponibile. Assicurati che sia avviato su localhost:1234")
                print("💡 Userò risposte template intelligenti")
        
        # Controlla in background per non bloccare l'avvio
        thread = threading.Thread(target=check_availability, daemon=True)
        thread.start()
    
    def set_scanned_path_context(self, scanned_path: str):
        """Set the most recently scanned path for context"""
        self.last_scanned_path = scanned_path
    
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
        
        if file_info.get("pdf_info"):
            pdf_info = file_info["pdf_info"]
            parts.append(f"PDF con {pdf_info['pages']} pagine")
        
        if file_info.get("docx_info"):
            docx_info = file_info["docx_info"]
            parts.append(f"Documento Word con {docx_info['paragraphs']} paragrafi")
        
        if file_info["type"] == "directory":
            parts.append(f"Directory con {file_info.get('file_count', 0)} file e {file_info.get('dir_count', 0)} sottodirectory")
        
        # NUOVA SEZIONE: Aggiungi contenuto estratto per ricerca semantica
        content_parts = []
        
        # Contenuto di file di testo, PDF, DOCX
        if file_info.get("content_preview"):
            content_preview = file_info["content_preview"]
            # Pulisci e limita il contenuto per l'embedding
            content_cleaned = content_preview.replace('\n', ' ').replace('\t', ' ')
            # Limita a 2000 caratteri per performance degli embeddings
            if len(content_cleaned) > 2000:
                content_cleaned = content_cleaned[:2000] + "..."
            content_parts.append(f"CONTENUTO: {content_cleaned}")
            print(f"🔍 Vettorizzando contenuto per {file_info.get('name', 'file')}: {len(content_cleaned)} caratteri")
        
        # Combina metadati e contenuto
        base_text = " | ".join(parts)
        if content_parts:
            return base_text + " | " + " | ".join(content_parts)
        else:
            return base_text
    
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
    
    async def chat_with_ai(self, user_message: str, conversation_history: List[Dict] = None, n_results: int = 50) -> Dict[str, Any]:
        """Chat conversazionale con AI usando solo LM Studio"""
        
        # Se LM Studio non è disponibile, restituisci errore
        if not self.lm_studio_available:
            return {
                "response": "❌ LM Studio non è disponibile. Avvialo su localhost:1234 e ricarica la pagina.",
                "context_used": 0,
                "relevant_files": [],
                "conversation_context": False
            }
        
        # Recupera sempre tutti i file per dare il contesto completo al modello
        all_files = await self._get_all_files_from_database()
        
        # Controlla se l'utente chiede il contenuto di un file specifico
        specific_file_content = await self._get_specific_file_content(user_message, all_files)
        
        # Prepara il contesto dettagliato per LM Studio
        context_parts = []
        if all_files:
            context_parts.append("=== INFORMAZIONI SUI FILE SCANSIONATI ===")
            
            # Statistiche generali
            total_files = len(all_files)
            total_size = sum(f["metadata"].get("size", 0) for f in all_files)
            total_human = self._format_size(total_size)
            context_parts.append(f"Totale: {total_files} file ({total_human})")
            
            # Percorso scansionato
            if self.last_scanned_path:
                context_parts.append(f"Cartella: {self.last_scanned_path}")
            
            # Se è stato richiesto un file specifico, aggiungi il suo contenuto
            if specific_file_content:
                context_parts.append(f"\n=== CONTENUTO DEL FILE RICHIESTO ===")
                context_parts.append(specific_file_content)
            
            # Primi file più grandi
            sorted_files = sorted(all_files, key=lambda x: x["metadata"].get("size", 0), reverse=True)
            context_parts.append("\nFile più grandi:")
            for i, f in enumerate(sorted_files[:10]):
                name = f['metadata']['name']
                size = f['metadata'].get('size_human', 'N/A')
                file_type = f['metadata'].get('type', 'file')
                context_parts.append(f"{i+1}. {name} - {size} ({file_type})")
            
            # Distribuzione per tipo
            types = {}
            for f in all_files:
                file_type = f['metadata'].get('type', 'unknown')
                types[file_type] = types.get(file_type, 0) + 1
            
            context_parts.append("\nDistribuzione per tipo:")
            for file_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"- {count} {file_type}")
        
        # Usa solo LM Studio per la risposta
        response = await self._generate_lm_studio_response(user_message, context_parts, conversation_history)
        
        return {
            "response": response if response else "❌ Errore nella comunicazione con LM Studio",
            "context_used": len(all_files),
            "relevant_files": sorted_files[:5] if all_files else [],
            "conversation_context": True
        }
    
    async def _get_all_files_from_database(self) -> List[Dict]:
        """Recupera file dal database filtrati per il contesto dell'ultima scansione"""
        try:
            # Ottieni tutti i documenti dal database ChromaDB
            all_results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            files = []
            if all_results and all_results["metadatas"]:
                for i, metadata in enumerate(all_results["metadatas"]):
                    files.append({
                        "metadata": metadata,
                        "document": all_results["documents"][i] if i < len(all_results["documents"]) else "",
                        "similarity_score": 1.0  # Non applicabile per recupero completo
                    })
            
            # Se abbiamo un contesto di ultima scansione, filtra per quel percorso
            if self.last_scanned_path and files:
                # Normalizza il percorso di contesto (espandi ~ se presente)
                context_path = os.path.expanduser(self.last_scanned_path)
                
                # Filtra i file che appartengono al percorso scansionato
                context_files = []
                for file_data in files:
                    file_path = file_data["metadata"].get("path", "")
                    # Controlla se il file è dentro il percorso di contesto
                    if file_path.startswith(context_path):
                        context_files.append(file_data)
                
                if context_files:
                    return context_files
            
            return files
            
        except Exception as e:
            print(f"Errore nel recupero di tutti i file: {e}")
            return []
    
    async def _generate_lm_studio_response(self, user_message: str, context_parts: List[str], conversation_history: List[Dict]) -> str:
        """Genera risposta usando LM Studio API"""
        try:
            # Prepara il contesto
            context_text = "\n".join(context_parts) if context_parts else ""
            
            # Costruisci i messaggi per l'API (senza system role per compatibilità Mistral)
            messages = []
            
            # Aggiungi storico conversazione se presente
            if conversation_history:
                for msg in conversation_history[-5:]:  # Ultimi 5 messaggi
                    role = "user" if msg.get("role") == "user" else "assistant"
                    if role in ["user", "assistant"]:  # Solo ruoli supportati
                        messages.append({
                            "role": role,
                            "content": msg.get("content", "")
                        })
            
            # Prepara il messaggio con istruzioni integrate
            user_prompt = f"""Sei un assistente intelligente per la gestione file. Rispondi in italiano in modo utile e conciso.

Contesto file: {context_text}

Domanda: {user_message}"""
            
            # Aggiungi il messaggio corrente con istruzioni integrate
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Chiamata API a LM Studio
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300
                }
                
                response = await client.post(
                    self.lm_studio_url,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"Risposta LM Studio non valida: {result}")
                        return "Errore: risposta non valida da LM Studio"
                else:
                    error_text = response.text if hasattr(response, 'text') else 'N/A'
                    print(f"LM Studio API error: {response.status_code} - {error_text}")
                    print(f"Payload inviato: {payload}")
                    return f"❌ Errore API LM Studio: {response.status_code}"
                    
        except Exception as e:
            print(f"Errore nella chiamata a LM Studio: {e}")
            self.lm_studio_available = False
            return f"❌ Errore connessione LM Studio: {str(e)}"
    
    
    async def _get_specific_file_content(self, user_message: str, all_files: List[Dict]) -> str:
        """Rileva se l'utente chiede il contenuto di un file specifico e lo restituisce"""
        user_message_lower = user_message.lower()
        
        # Parole chiave che indicano richiesta di contenuto
        content_keywords = ['leggi', 'mostra', 'contenuto', 'riassunto', 'riassumi', 'cosa dice', 'cosa contiene', 'apri']
        
        if not any(keyword in user_message_lower for keyword in content_keywords):
            return ""
        
        # Cerca il file menzionato nel messaggio
        for file_data in all_files:
            file_name = file_data["metadata"].get("name", "")
            
            # Controlla se il nome del file (o parte di esso) è nel messaggio
            file_name_parts = file_name.lower().replace('.pdf', '').replace('.txt', '').replace('.docx', '').split()
            
            # Se almeno 2 parole del nome del file sono nel messaggio
            matches = sum(1 for part in file_name_parts if len(part) > 2 and part in user_message_lower)
            
            if matches >= min(2, len(file_name_parts)) or file_name.lower() in user_message_lower:
                # File trovato! Restituisci il contenuto se disponibile
                content = file_data.get("document", "")
                if "CONTENUTO:" in content:
                    # Estrai solo la parte del contenuto
                    content_start = content.find("CONTENUTO:") + len("CONTENUTO:")
                    extracted_content = content[content_start:].strip()
                    
                    return f"FILE: {file_name}\n\nCONTENUTO ESTRATTO:\n{extracted_content[:8000]}{'...' if len(extracted_content) > 8000 else ''}"
                else:
                    return f"FILE: {file_name}\n\nIl file è stato trovato ma non contiene contenuto testuale estraibile."
        
        return ""

    def _format_size(self, size_bytes):
        """Formatta dimensione in bytes in formato leggibile"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
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