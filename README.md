# RAG System for Files Management 📁🤖

Un sistema avanzato di Retrieval-Augmented Generation (RAG) per l'analisi e gestione intelligente dei file del tuo sistema, con integrazione AI per chat conversazionale.

## ✨ Funzionalità Principali

### 📊 Analisi File Avanzata
- **Scansione ricorsiva** di directory con metadati dettagliati
- **Analisi contenuto** per diversi tipi di file (testo, immagini, PDF, Excel, etc.)
- **Calcolo dimensioni** con visualizzazione human-readable
- **Rilevamento tipo MIME** e categorizzazione automatica
- **Suggerimenti pulizia** per file temporanei, cache e duplicati

### 🔍 Sistema RAG con Vector Search
- **Embeddings semantici** con Sentence Transformers (all-MiniLM-L6-v2)
- **Database vettoriale ChromaDB** per ricerca veloce e accurata
- **Ricerca per similarità** con ranking di rilevanza
- **Filtri avanzati** per dimensione, tipo, estensione
- **Contesto persistente** per l'ultima cartella scansionata

### 💬 Chat AI Conversazionale
- **Integrazione LM Studio** per usare qualsiasi modello LLM locale
- **Risposte contestuali** basate sui file scansionati
- **Analisi intelligente** con suggerimenti personalizzati
- **Compatibilità estesa** con modelli Mistral, Llama, Qwen, ecc.
- **Supporto multilingua** (italiano/inglese)

### 📈 Visualizzazioni Dati
- **Grafico vettori 3D** con PCA per visualizzare la distribuzione dei file
- **Categorizzazione visuale** per tipo di file con colori distinti
- **Distribuzione tipi file** con grafici interattivi

## 🚀 Quick Start

### Prerequisiti
- Docker e Docker Compose
- **LM Studio** installato e configurato
- Oppure Python 3.11+ per esecuzione locale

### Installazione con Docker (Consigliato)

```bash
# Clona il repository
git clone https://github.com/MarcoSalmaso/RAG-System-for-Files-Management.git
cd RAG-System-for-Files-Management

# Avvia il sistema
docker-compose up --build

# Apri nel browser
# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
```

### Configurazione LM Studio

1. **Scarica e installa LM Studio**: https://lmstudio.ai
2. **Carica un modello** (es. Mistral 7B, Llama 3.2, Qwen):
   - Vai nella tab "Models" 
   - Cerca e scarica un modello compatibile
3. **Avvia il server locale**:
   - Vai nella tab "Local Server"
   - Carica il modello nella chat
   - Clicca "Start Server" su porta 1234
   - ✅ Verifica che appaia "Server running on http://localhost:1234"

### Installazione Locale

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (in un'altra finestra)
cd frontend-simple
python -m http.server 5173
```

## 📖 Utilizzo

### 1. Scansiona una Directory
- Vai alla tab **"Scansiona"**
- Inserisci il percorso (es. `~/Desktop` o `/Users/nome/Documents`)
- Clicca su "Analizza Directory"
- Il sistema indicizzerà tutti i file trovati

### 2. Chat Intelligente con LM Studio
- **Assicurati che LM Studio sia attivo** su localhost:1234
- Vai alla tab **"Chat AI"**
- Fai domande sui tuoi file:
  - "Quali sono i file più grandi?"
  - "Mostrami le cartelle più pesanti"
  - "Cosa posso eliminare per liberare spazio?"
  - "Dammi un riassunto della cartella"
  - "Trova tutti i file immagine"
- Il tuo modello LM Studio risponderà con contesto completo sui file!

### 3. Visualizza Grafico Vettori
- Vai alla tab **"Grafico Vettori"**
- Esplora la rappresentazione 3D dei tuoi file
- I colori indicano diverse categorie (immagini, documenti, codice, etc.)
- Passa il mouse sui punti per vedere i dettagli

## 🏗️ Architettura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │────▶│   ChromaDB      │
│   (HTML/JS)     │     │   (FastAPI)     │     │   (Vectors)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AI Models         │
                    │ - LM Studio API     │
                    │ - Sentence-Transform│
                    └─────────────────────┘
```

## 🛠️ Stack Tecnologico

### Backend
- **FastAPI**: Framework web moderno e veloce
- **ChromaDB**: Database vettoriale per embeddings
- **Sentence Transformers**: Generazione embeddings semantici
- **LM Studio Integration**: API REST per modelli LLM locali
- **HTTPX**: Client HTTP asincrono per LM Studio API
- **Python-Magic**: Rilevamento tipo MIME
- **PyPDF2, python-docx, openpyxl**: Parsing documenti
- **Pillow**: Analisi immagini

### Frontend
- **HTML5/CSS3/JavaScript**: Interfaccia responsive
- **Tailwind CSS**: Styling moderno
- **Chart.js**: Visualizzazione grafici
- **Three.js**: Rendering 3D vettori
- **Fetch API**: Comunicazione con backend

## 📋 API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/scan` | POST | Scansiona una directory |
| `/chat` | POST | Chat conversazionale AI |
| `/query` | POST | Query RAG sui file |
| `/vectors` | GET | Vettori per visualizzazione |
| `/file-types` | GET | Distribuzione tipi file |
| `/clear` | DELETE | Pulisce l'indice |

## 🔧 Configurazione

### Docker Compose
```yaml
services:
  backend:
    ports:
      - "8000:8000"
    volumes:
      - ./backend/chroma_db:/app/chroma_db  # Persistenza DB
      - /:/host_root:ro  # Accesso file sistema (read-only)

  frontend:
    ports:
      - "5173:80"
```

### Variabili Ambiente
- `CHROMA_DB_PATH`: Path database vettoriale (default: `./chroma_db`)
- `MODEL_CACHE_DIR`: Cache modelli AI (default: `~/.cache/huggingface`)
- `MAX_FILE_SIZE`: Dimensione max file analizzabile (default: 100MB)

## 📁 Struttura Progetto

```
RAG-System-for-Files-Management/
├── backend/
│   ├── main.py              # API FastAPI
│   ├── rag_system.py        # Core RAG + AI
│   ├── file_analyzer.py     # Analisi file
│   ├── requirements.txt     # Dipendenze Python
│   └── Dockerfile           
├── frontend-simple/
│   ├── index.html           # Interfaccia web
│   └── Dockerfile
├── docker-compose.yml       # Orchestrazione servizi
└── README.md               # Questo file
```

## 🚦 Limitazioni e Note

- **LM Studio richiesto**: Necessario avere LM Studio installato e attivo
- **Compatibilità modelli**: Testato con Mistral, Llama, Qwen (modelli senza ruolo "system")
- **Memoria**: RAM richiesta dipende dal modello caricato in LM Studio
- **File grandi**: File >100MB potrebbero rallentare l'indicizzazione
- **Percorsi Docker**: In Docker, i percorsi locali sono mappati su `/host_root`
- **Connettività**: Il container Docker deve poter accedere a `host.docker.internal:1234`

## 📝 License

MIT License - Vedi file [LICENSE](LICENSE) per dettagli

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) per il database vettoriale
- [LM Studio](https://lmstudio.ai/) per l'interfaccia AI locale
- [Hugging Face](https://huggingface.co/) per i modelli AI
- [Sentence Transformers](https://www.sbert.net/) per gli embeddings

---

Sviluppato con ❤️ per rendere la gestione file più intelligente e conversazionale.