import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes
import hashlib
from datetime import datetime
import psutil
import magic
from PIL import Image
import json

class FileAnalyzer:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        
    async def scan_directory(self, path: str, include_hidden: bool = False, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Scansiona una directory e restituisce metadati dettagliati sui file"""
        results = []
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        
        await self._scan_recursive(path_obj, results, include_hidden, max_depth, 0)
        return results
    
    async def _scan_recursive(self, path: Path, results: List[Dict[str, Any]], 
                            include_hidden: bool, max_depth: int, current_depth: int):
        """Scansione ricorsiva delle directory"""
        if current_depth > max_depth:
            return
            
        try:
            for item in path.iterdir():
                # Salta file nascosti se non richiesti
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                if item.is_file():
                    file_info = await self._analyze_file(item)
                    if file_info:
                        results.append(file_info)
                elif item.is_dir():
                    # Analizza la directory
                    dir_info = await self._analyze_directory(item)
                    results.append(dir_info)
                    
                    # Continua la scansione ricorsiva
                    await self._scan_recursive(item, results, include_hidden, max_depth, current_depth + 1)
                    
        except PermissionError:
            # Salta directory senza permessi
            pass
    
    async def _analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analizza un singolo file e restituisce i metadati"""
        try:
            stat = file_path.stat()
            
            # Informazioni base
            file_info = {
                "path": str(file_path),
                "name": file_path.name,
                "type": "file",
                "size": stat.st_size,
                "size_human": self._format_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "extension": file_path.suffix.lower(),
                "permissions": oct(stat.st_mode)[-3:],
            }
            
            # Tipo MIME
            try:
                mime_type = self.mime.from_file(str(file_path))
                file_info["mime_type"] = mime_type
            except:
                file_info["mime_type"] = "unknown"
            
            # Hash del file (solo per file piccoli < 10MB)
            if stat.st_size < 10 * 1024 * 1024:
                file_info["md5_hash"] = await self._calculate_hash(file_path)
            
            # Analisi specifica per tipo di file
            await self._analyze_file_content(file_path, file_info)
            
            # Suggerimenti per la pulizia
            file_info["cleanup_suggestions"] = await self._get_cleanup_suggestions(file_info)
            
            # Genera una descrizione intelligente del file analizzando anche il contenuto
            file_info["description"] = await self._generate_intelligent_description(file_path, file_info)
            
            return file_info
            
        except (OSError, PermissionError):
            return None
    
    def _get_directory_size_recursive(self, dir_path: Path) -> int:
        """Calcola la dimensione totale di una directory ricorsivamente"""
        total_size = 0
        try:
            for item in dir_path.iterdir():
                try:
                    if item.is_file():
                        total_size += item.stat().st_size
                    elif item.is_dir():
                        # Ricorsivamente calcola la dimensione delle sottodirectory
                        total_size += self._get_directory_size_recursive(item)
                except (PermissionError, OSError):
                    continue
        except (PermissionError, OSError):
            pass
        return total_size
    
    async def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analizza una directory"""
        try:
            stat = dir_path.stat()
            
            # Conta file e sottodirectory (solo primo livello per info)
            file_count = 0
            dir_count = 0
            
            try:
                for item in dir_path.iterdir():
                    if item.is_file():
                        file_count += 1
                    elif item.is_dir():
                        dir_count += 1
            except PermissionError:
                pass
            
            # Calcola la dimensione totale ricorsivamente
            total_size = self._get_directory_size_recursive(dir_path)
            
            dir_info = {
                "path": str(dir_path),
                "name": dir_path.name,
                "type": "directory",
                "size": total_size,  # Aggiungi anche il campo "size" per l'ordinamento
                "file_count": file_count,
                "dir_count": dir_count,
                "total_size": total_size,
                "size_human": self._format_size(total_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "permissions": oct(stat.st_mode)[-3:],
                "cleanup_suggestions": await self._get_directory_cleanup_suggestions(dir_path, file_count, total_size)
            }
            
            # Genera una descrizione intelligente per la directory
            dir_info["description"] = await self._generate_intelligent_directory_description(dir_info, dir_path)
            
            return dir_info
            
        except (OSError, PermissionError):
            return {
                "path": str(dir_path),
                "name": dir_path.name,
                "type": "directory",
                "error": "Permission denied"
            }
    
    async def _analyze_file_content(self, file_path: Path, file_info: Dict[str, Any]):
        """Analizza il contenuto specifico del file"""
        extension = file_path.suffix.lower()
        mime_type = file_info.get("mime_type", "")
        
        # Immagini
        if mime_type.startswith("image/"):
            try:
                with Image.open(file_path) as img:
                    file_info["image_info"] = {
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode
                    }
            except:
                pass
        
        # File di testo
        elif mime_type.startswith("text/") or extension in ['.txt', '.py', '.js', '.html', '.css', '.json', '.xml']:
            try:
                if file_info["size"] < 1024 * 1024:  # Max 1MB per analisi testo
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        file_info["text_info"] = {
                            "lines": len(content.split('\n')),
                            "chars": len(content),
                            "words": len(content.split())
                        }
            except:
                pass
    
    async def _calculate_hash(self, file_path: Path) -> str:
        """Calcola l'hash MD5 del file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def _format_size(self, size_bytes: int) -> str:
        """Formatta la dimensione in formato leggibile"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    async def _get_cleanup_suggestions(self, file_info: Dict[str, Any]) -> List[str]:
        """Genera suggerimenti per la pulizia del file"""
        suggestions = []
        
        # File molto grandi
        if file_info["size"] > 100 * 1024 * 1024:  # > 100MB
            suggestions.append("File molto grande - considera se è necessario")
        
        # File vecchi non accessi
        try:
            accessed = datetime.fromisoformat(file_info["accessed"])
            if (datetime.now() - accessed).days > 365:
                suggestions.append("Non accesso da oltre un anno")
        except:
            pass
        
        # File duplicati (basato su hash)
        # TODO: implementare controllo duplicati
        
        # File temporanei
        temp_extensions = ['.tmp', '.temp', '.bak', '.old', '.~']
        if file_info["extension"] in temp_extensions:
            suggestions.append("File temporaneo - probabilmente eliminabile")
        
        # File di cache
        cache_paths = ['cache', 'tmp', 'temp', '.cache']
        if any(cache in file_info["path"].lower() for cache in cache_paths):
            suggestions.append("File di cache - può essere eliminato")
        
        return suggestions
    
    async def _get_directory_cleanup_suggestions(self, dir_path: Path, file_count: int, total_size: int) -> List[str]:
        """Genera suggerimenti per la pulizia della directory"""
        suggestions = []
        
        if total_size > 1024 * 1024 * 1024:  # > 1GB
            suggestions.append("Directory molto grande")
        
        if file_count > 1000:
            suggestions.append("Molti file - considera l'organizzazione")
        
        # Directory comuni da pulire
        cleanup_dirs = ['Downloads', 'Trash', 'Cache', 'Logs']
        if any(cleanup in dir_path.name for cleanup in cleanup_dirs):
            suggestions.append("Directory spesso da pulire")
        
        return suggestions
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche del sistema"""
        disk_usage = psutil.disk_usage('/')
        memory = psutil.virtual_memory()
        
        return {
            "disk": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": (disk_usage.used / disk_usage.total) * 100,
                "total_human": self._format_size(disk_usage.total),
                "used_human": self._format_size(disk_usage.used),
                "free_human": self._format_size(disk_usage.free)
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "total_human": self._format_size(memory.total),
                "available_human": self._format_size(memory.available)
            }
        }
    
    def _generate_file_description(self, file_info: Dict[str, Any]) -> str:
        """Genera una descrizione intelligente basata sul contenuto e contesto del file"""
        ext = file_info.get("extension", "").lower()
        mime = file_info.get("mime_type", "")
        size = file_info.get("size", 0)
        name = file_info.get("name", "").replace(ext, "")  # Nome senza estensione
        path = file_info.get("path", "")
        
        # Analizza il nome del file per estrarre informazioni contestuali
        name_lower = name.lower()
        
        # Cerca pattern comuni nel nome del file
        description_parts = []
        
        # Pattern per date
        import re
        date_patterns = [
            (r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', 'del {2}/{1}/{0}'),
            (r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})', 'del {0}/{1}/{2}'),
            (r'(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)[-_]?(\d{4})', 'di {0} {1}'),
            (r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_]?(\d{4})', 'di {0} {1}'),
        ]
        
        for pattern, format_str in date_patterns:
            match = re.search(pattern, name_lower)
            if match:
                date_info = format_str.format(*match.groups())
                description_parts.append(date_info)
                break
        
        # Analizza il tipo di contenuto dal nome
        content_hints = {
            'budget': 'Budget',
            'bilancio': 'Bilancio',
            'fattura': 'Fattura',
            'invoice': 'Fattura',
            'ricevuta': 'Ricevuta',
            'receipt': 'Ricevuta',
            'contratto': 'Contratto',
            'contract': 'Contratto',
            'report': 'Report',
            'relazione': 'Relazione',
            'presentazione': 'Presentazione',
            'presentation': 'Presentazione',
            'curriculum': 'Curriculum',
            'cv': 'CV',
            'resume': 'CV',
            'lettera': 'Lettera',
            'letter': 'Lettera',
            'foto': 'Foto',
            'photo': 'Foto',
            'screenshot': 'Screenshot',
            'schermata': 'Screenshot',
            'backup': 'Backup',
            'copia': 'Copia',
            'progetto': 'Progetto',
            'project': 'Progetto',
            'documento': 'Documento',
            'document': 'Documento',
            'manuale': 'Manuale',
            'manual': 'Manuale',
            'guida': 'Guida',
            'guide': 'Guida',
            'tutorial': 'Tutorial',
            'esercizio': 'Esercizio',
            'exercise': 'Esercizio',
            'compito': 'Compito',
            'homework': 'Compito',
            'appunti': 'Appunti',
            'notes': 'Note',
            'meeting': 'Riunione',
            'riunione': 'Riunione',
            'verbale': 'Verbale',
            'ricetta': 'Ricetta',
            'recipe': 'Ricetta',
            'lista': 'Lista',
            'list': 'Lista',
            'todo': 'Lista TODO',
            'password': 'Password',
            'config': 'Configurazione',
            'settings': 'Impostazioni',
            'log': 'Log',
            'error': 'Errori',
            'download': 'Download',
            'temp': 'Temporaneo',
            'tmp': 'Temporaneo',
            'draft': 'Bozza',
            'bozza': 'Bozza',
            'finale': 'Versione finale',
            'final': 'Versione finale',
            'v1': 'Versione 1',
            'v2': 'Versione 2',
            'nuovo': 'Nuovo',
            'new': 'Nuovo',
            'old': 'Vecchio',
            'vecchio': 'Vecchio',
        }
        
        # Trova il tipo di contenuto principale
        main_content = None
        for keyword, content_type in content_hints.items():
            if keyword in name_lower:
                main_content = content_type
                break
        
        # Estrai informazioni aggiuntive dal nome
        # Cerca nomi propri, aziende, progetti
        words = name.split('_')
        if not words or len(words) == 1:
            words = name.split('-')
        if not words or len(words) == 1:
            words = name.split(' ')
        
        # Filtra parole comuni e mantieni quelle significative
        stop_words = {'il', 'la', 'di', 'da', 'per', 'con', 'su', 'the', 'of', 'for', 'and', 'to', 'in', 'on', 'at', 'by', 'with'}
        meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in stop_words and w.lower() not in content_hints]
        
        # Costruisci la descrizione basata sul tipo di file
        if ext == '.pdf':
            if main_content:
                desc = f"{main_content}"
                if meaningful_words:
                    desc += f" {' '.join(meaningful_words[:3])}"
                if description_parts:
                    desc += f" {' '.join(description_parts)}"
            else:
                desc = f"PDF"
                if meaningful_words:
                    desc += f": {' '.join(meaningful_words[:3])}"
                if description_parts:
                    desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.doc', '.docx']:
            if main_content:
                desc = f"{main_content}"
            else:
                desc = "Documento Word"
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.xls', '.xlsx']:
            if 'budget' in name_lower or 'bilancio' in name_lower:
                desc = "Foglio calcolo budget"
            elif 'spese' in name_lower or 'expenses' in name_lower:
                desc = "Foglio calcolo spese"
            elif main_content:
                desc = f"Foglio calcolo {main_content}"
            else:
                desc = "Foglio Excel"
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            img_info = file_info.get("image_info", {})
            if 'screenshot' in name_lower or 'schermata' in name_lower:
                desc = "Screenshot"
            elif 'foto' in name_lower or 'photo' in name_lower:
                desc = "Foto"
            elif 'logo' in name_lower:
                desc = "Logo"
            elif 'icona' in name_lower or 'icon' in name_lower:
                desc = "Icona"
            elif main_content:
                desc = f"Immagine {main_content}"
            else:
                desc = "Immagine"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if img_info:
                desc += f" ({img_info.get('width', '')}x{img_info.get('height', '')}px)"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            if 'film' in name_lower or 'movie' in name_lower:
                desc = "Film"
            elif 'video' in name_lower:
                desc = "Video"
            elif 'tutorial' in name_lower:
                desc = "Video tutorial"
            elif 'presentazione' in name_lower or 'presentation' in name_lower:
                desc = "Video presentazione"
            else:
                desc = "Video"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
            if 'musica' in name_lower or 'music' in name_lower or 'song' in name_lower:
                desc = "Musica"
            elif 'podcast' in name_lower:
                desc = "Podcast"
            elif 'audio' in name_lower:
                desc = "Audio"
            elif 'registrazione' in name_lower or 'recording' in name_lower:
                desc = "Registrazione"
            else:
                desc = "Audio"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext == '.txt':
            txt_info = file_info.get("text_info", {})
            if main_content:
                desc = f"Testo {main_content}"
            elif 'readme' in name_lower:
                desc = "File README"
            elif 'license' in name_lower:
                desc = "Licenza"
            elif 'changelog' in name_lower:
                desc = "Changelog"
            else:
                desc = "Testo"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if txt_info and txt_info.get('lines', 0) > 10:
                desc += f" ({txt_info.get('lines', 0)} righe)"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        elif ext in ['.py', '.js', '.java', '.cpp', '.c']:
            txt_info = file_info.get("text_info", {})
            lang_map = {'.py': 'Python', '.js': 'JavaScript', '.java': 'Java', '.cpp': 'C++', '.c': 'C'}
            lang = lang_map.get(ext, 'Codice')
            
            if 'test' in name_lower:
                desc = f"Test {lang}"
            elif 'main' in name_lower:
                desc = f"{lang} principale"
            elif 'config' in name_lower or 'settings' in name_lower:
                desc = f"Configurazione {lang}"
            elif 'utils' in name_lower or 'utility' in name_lower:
                desc = f"Utility {lang}"
            elif 'model' in name_lower:
                desc = f"Model {lang}"
            elif 'view' in name_lower:
                desc = f"View {lang}"
            elif 'controller' in name_lower:
                desc = f"Controller {lang}"
            else:
                desc = f"Script {lang}"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:2])}"
            if txt_info and txt_info.get('lines', 0) > 0:
                desc += f" ({txt_info.get('lines', 0)} righe)"
        
        elif ext in ['.zip', '.rar', '.tar', '.gz', '.7z']:
            if 'backup' in name_lower:
                desc = "Backup compresso"
            elif 'progetto' in name_lower or 'project' in name_lower:
                desc = "Progetto compresso"
            elif main_content:
                desc = f"Archivio {main_content}"
            else:
                desc = "Archivio"
            
            if meaningful_words:
                desc += f": {' '.join(meaningful_words[:3])}"
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        else:
            # Descrizione generica basata sul nome
            if main_content:
                desc = main_content
            elif meaningful_words:
                desc = ' '.join(meaningful_words[:3])
            else:
                desc = f"File {ext[1:].upper() if ext else 'generico'}"
            
            if description_parts:
                desc += f" {' '.join(description_parts)}"
        
        # Aggiungi informazioni sul percorso se significative
        path_parts = path.split('/')
        if len(path_parts) > 2:
            parent_dir = path_parts[-2]
            if parent_dir.lower() in ['desktop', 'downloads', 'documents', 'documenti']:
                pass  # Non aggiungere, è ovvio
            elif any(proj in parent_dir.lower() for proj in ['progetto', 'project', 'lavoro', 'work', 'clienti', 'client']):
                desc = f"[{parent_dir}] {desc}"
        
        return desc
    
    async def _generate_intelligent_description(self, file_path: Path, file_info: Dict[str, Any]) -> str:
        """Genera una descrizione veramente intelligente analizzando il contenuto del file"""
        ext = file_info.get("extension", "").lower()
        mime = file_info.get("mime_type", "")
        size = file_info.get("size", 0)
        name = file_info.get("name", "")
        
        # Prima ottieni una descrizione base dal nome
        base_desc = self._generate_file_description(file_info)
        
        # Ora arricchisci con l'analisi del contenuto
        enhanced_desc = base_desc
        
        try:
            # Per file di testo, leggi le prime righe per capire il contenuto
            if mime and mime.startswith("text/") or ext in ['.txt', '.md', '.log', '.csv', '.json', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.c']:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_lines = []
                        for i, line in enumerate(f):
                            if i >= 10:  # Leggi solo le prime 10 righe
                                break
                            first_lines.append(line.strip())
                    
                    content_preview = ' '.join(first_lines)[:500]  # Primi 500 caratteri
                    
                    # Analizza il contenuto per estrarre informazioni
                    if ext == '.md' or ext == '.txt':
                        if content_preview.startswith('#'):
                            # È un markdown con titolo
                            title = content_preview.split('\n')[0].replace('#', '').strip()
                            enhanced_desc = f"Documento: \"{title[:50]}...\"" if len(title) > 50 else f"Documento: \"{title}\""
                        elif 'README' in name.upper():
                            enhanced_desc = f"Documentazione del progetto che spiega {content_preview[:100]}..."
                        elif any(word in content_preview.lower() for word in ['todo', 'task', 'fixme']):
                            enhanced_desc = f"Lista di task/TODO con elementi da completare"
                        elif any(word in content_preview.lower() for word in ['error', 'exception', 'warning']):
                            enhanced_desc = f"Log con errori o avvisi del sistema"
                        elif content_preview:
                            # Estrai una frase significativa
                            first_sentence = content_preview.split('.')[0][:100]
                            if first_sentence:
                                enhanced_desc = f"Testo che tratta di: {first_sentence}..."
                    
                    elif ext == '.json':
                        import json
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            if isinstance(data, dict):
                                keys = list(data.keys())[:5]
                                enhanced_desc = f"Dati JSON con campi: {', '.join(keys)}"
                                if 'config' in name.lower() or 'settings' in name.lower():
                                    enhanced_desc = f"Configurazione con parametri: {', '.join(keys[:3])}"
                                elif 'package' in name.lower():
                                    if 'dependencies' in data:
                                        deps_count = len(data.get('dependencies', {}))
                                        enhanced_desc = f"Package.json con {deps_count} dipendenze"
                            elif isinstance(data, list):
                                enhanced_desc = f"Array JSON con {len(data)} elementi"
                        except:
                            pass
                    
                    elif ext in ['.py', '.js', '.java', '.cpp', '.c']:
                        # Analizza codice sorgente
                        imports = []
                        classes = []
                        functions = []
                        
                        for line in first_lines:
                            if ext == '.py':
                                if line.startswith('import ') or line.startswith('from '):
                                    module = line.split()[1].split('.')[0]
                                    imports.append(module)
                                elif line.startswith('class '):
                                    class_name = line.split()[1].split('(')[0].rstrip(':')
                                    classes.append(class_name)
                                elif line.startswith('def '):
                                    func_name = line.split()[1].split('(')[0]
                                    functions.append(func_name)
                            elif ext in ['.js', '.ts']:
                                if 'import ' in line or 'require(' in line:
                                    imports.append("modules")
                                elif 'function ' in line or 'const ' in line and '=>' in line:
                                    functions.append("functions")
                                elif 'class ' in line:
                                    classes.append("classes")
                        
                        desc_parts = []
                        if classes:
                            desc_parts.append(f"Definisce classe {classes[0]}")
                        elif functions:
                            desc_parts.append(f"Implementa funzione {functions[0]}")
                        if imports:
                            desc_parts.append(f"usa {', '.join(imports[:2])}")
                        
                        if desc_parts:
                            enhanced_desc = f"Codice che {' e '.join(desc_parts)}"
                    
                    elif ext == '.csv':
                        # Per file CSV, leggi l'header
                        if first_lines:
                            headers = first_lines[0].split(',')
                            enhanced_desc = f"Dati tabellari con colonne: {', '.join(headers[:4])}"
                            if len(first_lines) > 1:
                                enhanced_desc += f" ({len(first_lines)-1}+ righe)"
                    
                    elif ext == '.html':
                        if '<title>' in content_preview:
                            import re
                            title_match = re.search(r'<title>(.*?)</title>', content_preview, re.IGNORECASE)
                            if title_match:
                                title = title_match.group(1)
                                enhanced_desc = f"Pagina web: \"{title}\""
                        elif 'bootstrap' in content_preview.lower():
                            enhanced_desc = "Pagina HTML con framework Bootstrap"
                        elif 'react' in content_preview.lower():
                            enhanced_desc = "Applicazione React"
                    
                    elif ext == '.css':
                        if 'bootstrap' in content_preview:
                            enhanced_desc = "Stili CSS Bootstrap personalizzati"
                        elif '@media' in content_preview:
                            enhanced_desc = "Stili CSS responsive con media queries"
                        else:
                            selectors = len([line for line in first_lines if '{' in line])
                            enhanced_desc = f"Foglio di stile con {selectors}+ regole CSS"
                    
                except Exception as e:
                    pass
            
            # Per file binari, aggiungi informazioni basate sui metadati
            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                img_info = file_info.get("image_info", {})
                created = file_info.get("created", "")
                
                if img_info:
                    width = img_info.get("width", 0)
                    height = img_info.get("height", 0)
                    
                    # Determina il tipo di immagine dalla risoluzione
                    if width > 3000 or height > 3000:
                        enhanced_desc = f"Foto ad alta risoluzione {width}x{height}px, probabilmente da fotocamera professionale"
                    elif width == 1920 and height == 1080:
                        enhanced_desc = f"Immagine Full HD 1920x1080, possibile screenshot o wallpaper"
                    elif width == height:
                        enhanced_desc = f"Immagine quadrata {width}x{height}px, possibile foto profilo o icona"
                    elif width < 200 and height < 200:
                        enhanced_desc = f"Thumbnail o icona piccola {width}x{height}px"
                    else:
                        aspect_ratio = width / height if height > 0 else 1
                        if aspect_ratio > 1.5:
                            enhanced_desc = f"Immagine panoramica {width}x{height}px"
                        else:
                            enhanced_desc = f"Immagine standard {width}x{height}px"
                
                # Aggiungi info sulla data se significativa
                if created:
                    try:
                        from datetime import datetime
                        created_date = datetime.fromisoformat(created)
                        days_old = (datetime.now() - created_date).days
                        if days_old == 0:
                            enhanced_desc += ", creata oggi"
                        elif days_old == 1:
                            enhanced_desc += ", creata ieri"
                        elif days_old < 7:
                            enhanced_desc += f", creata {days_old} giorni fa"
                        elif days_old < 30:
                            enhanced_desc += f", creata {days_old // 7} settimane fa"
                        elif days_old > 365:
                            enhanced_desc += f", vecchia di {days_old // 365} anni"
                    except:
                        pass
            
            # Per PDF, cerca di capire il tipo dal nome e dimensione
            elif ext == '.pdf':
                pages_estimate = size // (100 * 1024)  # Stima rozza: 100KB per pagina
                if pages_estimate > 100:
                    enhanced_desc = f"Documento PDF corposo (~{pages_estimate} pagine), probabilmente un libro o manuale"
                elif pages_estimate > 20:
                    enhanced_desc = f"Documento PDF medio (~{pages_estimate} pagine), possibile report o presentazione"
                elif pages_estimate > 5:
                    enhanced_desc = f"Documento PDF di {pages_estimate} pagine circa"
                else:
                    enhanced_desc = f"Documento PDF breve (poche pagine)"
                
                # Aggiungi context dal nome
                if any(word in name.lower() for word in ['cv', 'curriculum', 'resume']):
                    enhanced_desc = f"Curriculum Vitae in PDF"
                elif 'invoice' in name.lower() or 'fattura' in name.lower():
                    enhanced_desc = f"Fattura o documento fiscale in PDF"
                elif 'contract' in name.lower() or 'contratto' in name.lower():
                    enhanced_desc = f"Contratto o documento legale in PDF"
            
            # Per archivi compressi
            elif ext in ['.zip', '.rar', '.tar', '.gz', '.7z']:
                size_mb = size / (1024 * 1024)
                if size_mb > 1000:
                    enhanced_desc = f"Archivio molto grande ({self._format_size(size)}), probabilmente backup completo"
                elif size_mb > 100:
                    enhanced_desc = f"Archivio sostanzioso ({self._format_size(size)}), possibile progetto o collezione"
                else:
                    enhanced_desc = f"Archivio compresso ({self._format_size(size)})"
            
            # Per file di Office
            elif ext in ['.docx', '.doc']:
                if size > 10 * 1024 * 1024:
                    enhanced_desc = "Documento Word con molte immagini o contenuti multimediali"
                elif size > 1 * 1024 * 1024:
                    enhanced_desc = "Documento Word corposo, probabilmente con formattazione ricca"
                else:
                    enhanced_desc = "Documento Word standard"
            
            elif ext in ['.xlsx', '.xls']:
                if size > 5 * 1024 * 1024:
                    enhanced_desc = "Foglio Excel con molti dati, probabilmente database o analisi complessa"
                elif size > 500 * 1024:
                    enhanced_desc = "Foglio Excel con dati strutturati e possibili formule"
                else:
                    enhanced_desc = "Foglio Excel semplice"
                    
        except Exception as e:
            # In caso di errore, usa la descrizione base
            pass
        
        return enhanced_desc
    
    async def _generate_intelligent_directory_description(self, dir_info: Dict[str, Any], dir_path: Path) -> str:
        """Genera una descrizione intelligente per una directory analizzando contenuto e contesto"""
        
        # Prima ottieni una descrizione base
        base_desc = self._generate_directory_description(dir_info, dir_path)
        
        # Poi arricchisci analizzando il contenuto
        return base_desc
    
    def _generate_directory_description(self, dir_info: Dict[str, Any], dir_path: Path) -> str:
        """Genera una descrizione per una directory basata sul contenuto"""
        name = dir_info.get("name", "")
        file_count = dir_info.get("file_count", 0)
        dir_count = dir_info.get("dir_count", 0)
        total_size = dir_info.get("total_size", 0)
        path = str(dir_path)
        
        # Analizza i contenuti per determinare il tipo di directory
        try:
            # Conta i tipi di file principali
            file_types = {}
            items = list(dir_path.iterdir())[:50]  # Analizza max 50 elementi per velocità
            for item in items:
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext:
                        file_types[ext] = file_types.get(ext, 0) + 1
        except:
            file_types = {}
        
        # Descrizioni basate sul nome della directory
        name_lower = name.lower()
        
        if name_lower in ['desktop', 'scrivania']:
            return f"Desktop con {file_count} file e {dir_count} cartelle"
        elif name_lower == 'downloads' or name_lower == 'download':
            return f"Cartella download con {file_count} file scaricati"
        elif name_lower == 'documents' or name_lower == 'documenti':
            return f"Documenti personali ({file_count} file)"
        elif name_lower in ['images', 'pictures', 'immagini', 'foto']:
            return f"Raccolta immagini ({file_count} file)"
        elif name_lower in ['music', 'musica']:
            return f"Libreria musicale ({file_count} brani)"
        elif name_lower in ['videos', 'video', 'movies', 'film']:
            return f"Collezione video ({file_count} file)"
        elif name_lower == 'applications' or name_lower == 'applicazioni':
            return f"Cartella applicazioni ({file_count} app)"
        elif name_lower == 'library' or name_lower == 'libreria':
            return "Libreria di sistema"
        elif name_lower == 'system':
            return "File di sistema"
        elif name_lower == 'cache':
            return f"Cache di sistema ({self._format_size(total_size)})"
        elif name_lower == 'temp' or name_lower == 'tmp':
            return f"File temporanei ({self._format_size(total_size)})"
        elif name_lower == 'backup' or name_lower == 'backups':
            return f"Backup ({file_count} file, {self._format_size(total_size)})"
        elif 'node_modules' in name_lower:
            return f"Dipendenze Node.js ({file_count} pacchetti)"
        elif '.git' == name:
            return "Repository Git"
        elif name_lower == 'src' or name_lower == 'source':
            return f"Codice sorgente ({file_count} file)"
        elif name_lower == 'bin':
            return "File binari ed eseguibili"
        elif name_lower == 'dist' or name_lower == 'build':
            return f"File compilati ({self._format_size(total_size)})"
        elif name_lower == 'public' or name_lower == 'static':
            return "File pubblici/statici web"
        elif name_lower == 'assets':
            return f"Risorse ({file_count} file)"
        elif name_lower == 'logs':
            return f"File di log ({file_count} file)"
        elif name_lower in ['test', 'tests', '__tests__']:
            return "Test del codice"
        elif name.startswith('.'):
            return f"Directory nascosta di configurazione"
        
        # Analizza in base ai tipi di file predominanti
        if file_types:
            most_common_ext = max(file_types.items(), key=lambda x: x[1])
            ext, count = most_common_ext
            
            if ext in ['.jpg', '.jpeg', '.png', '.gif']:
                return f"Cartella con {count}+ immagini"
            elif ext in ['.mp3', '.wav', '.flac']:
                return f"Cartella con {count}+ file audio"
            elif ext in ['.mp4', '.avi', '.mov']:
                return f"Cartella con {count}+ video"
            elif ext in ['.pdf', '.doc', '.docx', '.txt']:
                return f"Cartella con {count}+ documenti"
            elif ext in ['.py', '.js', '.java', '.cpp']:
                return f"Progetto di programmazione ({count}+ file di codice)"
        
        # Descrizione generica
        if file_count == 0 and dir_count == 0:
            return "Cartella vuota"
        elif file_count > 0 and dir_count == 0:
            return f"Contiene {file_count} file ({self._format_size(total_size)})"
        elif file_count == 0 and dir_count > 0:
            return f"Contiene {dir_count} sottocartelle"
        else:
            return f"Contiene {file_count} file e {dir_count} cartelle ({self._format_size(total_size)})"