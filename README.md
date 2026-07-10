# AI Agent Hub

**LangGraph + LangChain + Streamlit**

A production-grade AI Agent application with two advanced agent patterns:

| Module | Technology | Purpose |
|--------|-----------|---------|
| SQL Agent | LangGraph + LangChain | Natural language to SQL query engine |
| RAG Agent | LangGraph + ChromaDB | Semantic search + hybrid retrieval |

## Quick Start

### 1. Prerequisites
- Python 3.10+
- OpenAI API key (get one at https://platform.openai.com)

### 2. Setup
```bash
cd rag-agent-demo
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
python data/init_db.py
streamlit run app.py
```
Or double-click `run.bat` on Windows.

### 3. Access
Open http://localhost:8501

## Deployment Options

### Streamlit Community Cloud (Free)
1. Push to a GitHub repository
2. Go to share.streamlit.io and connect your repo
3. Add OPENAI_API_KEY in Streamlit Secrets

### Hugging Face Spaces (Free)
1. Create a Space at huggingface.co/spaces
2. Select Streamlit SDK
3. Upload files and set OPENAI_API_KEY in secrets

## Sample Queries

### SQL Agent
- Show me all parameters for carline E371
- What lighting parameters does DX11 have?
- List parameters with data type FLOAT
- Find all infotainment parameters

### RAG Agent
- E371 daytime running light parameters
- DX11 steering or traction control
- Network and communication params
- Braking system parameters

## Tech Stack

LangGraph (agent workflows), LangChain (LLM integration), OpenAI (LLM + embeddings),
ChromaDB (vector store), SQLite/PostgreSQL (structured data), Streamlit (UI)

## Database Schema

parameter_tables_parameter stores vehicle ECU parameters:
carline (TEXT), parameter_name (TEXT), parameter_group (TEXT),
data_type (TEXT), min/max_value (REAL), unit (TEXT), description (TEXT)

## License

MIT

Built for resume portfolio - AI Agent architectures with LangGraph & LangChain
