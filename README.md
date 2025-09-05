# RAG System - File Analyzer

Un sistema RAG (Retrieval-Augmented Generation) per analizzare e interrogare i file sul tuo computer locale utilizzando Docker.

## Funzionalità

- 🔍 **Scansione File**: Analizza ricorsivamente directory e file
- 💬 **Chat Intelligente**: Fai domande sui tuoi file utilizzando query in linguaggio naturale  
- 📊 **Visualizzazione Vettoriale**: Visualizza i documenti come punti in uno spazio 2D basato sulla loro similarità
- 📁 **Analisi Contenuto**: Genera descrizioni intelligenti basate sul contenuto dei file
- 🗂️ **Ordinamento per Dimensione**: Visualizza file e cartelle ordinati dal più grande al più piccolo

## Stack Tecnologico

### Backend
- **FastAPI**: Framework web Python per API REST
- **ChromaDB**: Database vettoriale per memorizzare embeddings
- **Sentence Transformers**: Modello per generare embeddings di testo (all-MiniLM-L6-v2)
- **Python Libraries**: pandas, numpy, scikit-learn, pillow, pypdf2, python-docx

### Frontend  
- **HTML/JavaScript**: Interfaccia web semplice e responsive
- **Tailwind CSS**: Framework CSS per lo styling
- **Chart.js**: Libreria per visualizzazioni grafiche
- **Canvas API**: Per la visualizzazione dei vettori

## Installazione

### Prerequisiti
- Docker e Docker Compose installati
- Almeno 2GB di RAM disponibile

### Setup

1. Clona il repository:
```bash
git clone <repository-url>
cd RAG-System
```

2. Avvia i container Docker:
```bash
docker-compose up --build
```

3. Accedi all'interfaccia web:
```
http://localhost:5173
```

## Utilizzo

### 1. Scansione Directory
- Vai alla tab "Scansiona Directory"
- Inserisci il percorso della directory da analizzare (es: `~/Desktop` o `/percorso/completo`)
- Clicca "Avvia Scansione"
- I risultati saranno mostrati ordinati per dimensione

### 2. Domande AI
- Vai alla tab "Domande AI"
- Scrivi la tua domanda nella casella di testo
- Esempi di domande:
  - "Quali sono i file più grandi?"
  - "Mostrami le immagini"
  - "Trova documenti PDF"
  - "Quali file posso eliminare per liberare spazio?"
- I risultati includeranno file rilevanti con punteggio di similarità

### 3. Visualizzazione Vettoriale
- Vai alla tab "Visualizzazione Vettori"
- Clicca "Aggiorna" per caricare la visualizzazione
- I punti rappresentano i file nel database
- Colori diversi indicano categorie diverse (immagini, documenti, codice, ecc.)
- Passa il mouse sui punti per vedere i dettagli del file

## Configurazione

### Percorsi
Il sistema monta automaticamente la home directory dell'utente. Per modificare questo comportamento, edita `docker-compose.yml`:

```yaml
volumes:
  - ${HOME}:/host_users:ro  # Modifica ${HOME} con il percorso desiderato
```

### Porte
- Backend API: porta 8000
- Frontend Web: porta 5173

Per cambiare le porte, modifica `docker-compose.yml`.

## API Endpoints

- `POST /scan`: Scansiona una directory
- `POST /query`: Esegue una query RAG
- `GET /vectors`: Ottiene i vettori per la visualizzazione
- `DELETE /clear`: Pulisce l'indice del database

## Struttura Progetto

```
RAG-System/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py              # FastAPI app principale
│   ├── file_analyzer.py     # Logica di analisi file
│   └── rag_system.py        # Sistema RAG con ChromaDB
├── frontend-simple/
│   ├── Dockerfile
│   ├── nginx.conf
│   └── index.html           # Interfaccia web
├── docker-compose.yml
└── README.md
```

## Sicurezza

- I file sono montati in modalità **read-only** nel container Docker
- Il sistema non modifica o elimina mai i file sul tuo computer
- Il pulsante "Pulisci Indice" rimuove solo i dati dal database vettoriale, non i file reali

## Troubleshooting

### Il container non si avvia
```bash
docker-compose down
docker-compose up --build
```

### Errore di permessi
Assicurati che Docker abbia i permessi per accedere alla directory che vuoi scansionare.

### Database corrotto
```bash
docker-compose down
docker volume rm rag-system_chroma_data
docker-compose up
```

## License

MIT License