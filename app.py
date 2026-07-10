"""
AI Agent Hub — RAG Agent & SQL Agent Demo
Built with LangGraph + LangChain + Streamlit

Architecture:
  ┌─ SQL Agent ──────────────────────────────────────────────────┐
  │  User Question → [LLM generates SQL] → [executes on SQLite]  │
  │                  → [LLM formats answer] → Response            │
  └──────────────────────────────────────────────────────────────┘
  ┌─ RAG Agent ──────────────────────────────────────────────────┐
  │  User Question → [retrieve from ChromaDB] → [LLM generates   │
  │                  answer with context] → Response              │
  └──────────────────────────────────────────────────────────────┘
"""

import streamlit as st
import os
import sqlite3
import json
from dotenv import load_dotenv
from typing import TypedDict, List

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "vehicle_params.db")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma_db")

def _get_config(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

LLM_API_KEY = _get_config("OPENAI_API_KEY")
LLM_BASE_URL = _get_config("OPENAI_BASE_URL")
LLM_MODEL = _get_config("LLM_MODEL", "gpt-4o-mini")

EMBEDDING_API_KEY = _get_config("OPENAI_API_KEY")
EMBEDDING_BASE_URL = _get_config("EMBEDDING_BASE_URL")
EMBEDDING_MODEL = _get_config("EMBEDDING_MODEL", "text-embedding-3-small")

TABLE_SCHEMA_PROMPT = """
Table: vehicle_parameters
Columns:
- id (INTEGER, PK): Auto-increment ID
- brand (TEXT): Vehicle brand (e.g. BMW, Tesla, VW, NIO, BYD, Mercedes)
- model (TEXT): Vehicle model (e.g. i4, Model 3, ID.4, ET5, Seal, EQS)
- carline (TEXT): Platform/carline code (e.g. G26, Highland, E371, NT2.0, E321, V297)
- system (TEXT): ECU system (BCM, GW, EPS, IPU, HU, AC, ESP)
- param_name (TEXT): Parameter name
- param_group (TEXT): Functional group (Lighting, Chassis, Network, Infotainment, HVAC)
- data_type (TEXT): Data type (BOOLEAN, UINT8, UINT16, UINT32, FLOAT)
- unit (TEXT): Measurement unit
- min_val (REAL): Minimum value
- max_val (REAL): Maximum value
- default_val (TEXT): Default value
- description (TEXT): Parameter description
"""

SAMPLE_QUERIES = {
    "SQL Agent": [
        "Show all BMW parameters",
        "List lighting parameters for carline E371",
        "How many parameters does Tesla have?",
        "What parameters have FLOAT data type?",
        "Show me all infotainment params",
    ],
    "RAG Agent": [
        "Tell me about DRL settings for BMW",
        "What braking or chassis parameters exist?",
        "Explain network and communication parameters",
        "Show me lighting-related parameters",
        "What infotainment parameters are available?",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Agent Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Cached Resources
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner="🤖 Connecting to LLM...")
def get_llm():
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=LLM_MODEL, api_key=LLM_API_KEY, temperature=0)
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return ChatOpenAI(**kwargs)


@st.cache_resource(show_spinner="🔤 Loading embeddings...")
def get_embeddings():
    # Try remote embedding API first (OpenAI / DeepSeek compatible)
    if EMBEDDING_API_KEY and EMBEDDING_API_KEY != "sk-your-api-key-here":
        try:
            from langchain_openai import OpenAIEmbeddings
            kwargs = dict(model=EMBEDDING_MODEL, api_key=EMBEDDING_API_KEY)
            if EMBEDDING_BASE_URL:
                kwargs["base_url"] = EMBEDDING_BASE_URL
            emb = OpenAIEmbeddings(**kwargs)
            emb.embed_query("test")
            return emb
        except Exception:
            pass
    # Fallback: ChromaDB built-in ONNX embedding (free, no API key needed)
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class OnnxEmbeddings:
        def __init__(self):
            self._ef = ONNXMiniLM_L6_V2()
        def embed_query(self, text):
            return [float(v) for v in self._ef([text])[0]]
        def embed_documents(self, texts):
            return [[float(v) for v in vec] for vec in self._ef(texts)]

    return OnnxEmbeddings()

@st.cache_resource(show_spinner="📚 Loading vector database...")
def get_vector_store():
    """
    Load ChromaDB from disk, or build it from SQLite data if not found.
    """
    from langchain_community.vectorstores import Chroma

    if os.path.exists(CHROMA_PATH):
        try:
            return Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embeddings(),
                collection_name="vehicle_params",
            )
        except Exception:
            pass  # fall through to rebuild

    # Build from scratch using existing SQLite data
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_parameters")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    documents, metadatas, ids = [], [], []
    for row in rows:
        record = dict(zip(columns, row))
        text = (
            f"Brand: {record['brand']} | Model: {record['model']} "
            f"| Carline: {record['carline']} | System: {record['system']} "
            f"| Parameter: {record['param_name']} | Group: {record['param_group']} "
            f"| Type: {record['data_type']} | Unit: {record['unit']} "
            f"| Min: {record['min_val']} | Max: {record['max_val']} "
            f"| Default: {record['default_val']} "
            f"| Description: {record['description']}"
        )
        documents.append(text)
        metadatas.append({
            "brand": record["brand"],
            "model": record["model"],
            "system": record["system"],
            "param_name": record["param_name"],
            "param_group": record["param_group"],
        })
        ids.append(str(record["id"]))

    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=get_embeddings(),
        metadatas=metadatas,
        ids=ids,
        persist_directory=CHROMA_PATH,
        collection_name="vehicle_params",
    )
    # Cloud ephemeral: no persist needed
    return vector_store


# ═══════════════════════════════════════════════════════════════════════════
# Database Helpers
# ═══════════════════════════════════════════════════════════════════════════


def run_sql(sql: str) -> dict:
    """Execute SQL and return {'success': bool, 'data': str, 'error': str}."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        result = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return {"success": True, "data": json.dumps(result, ensure_ascii=False, indent=2), "error": ""}
    except Exception as e:
        return {"success": False, "data": "", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# SQL Agent — LangGraph Definition
# ═══════════════════════════════════════════════════════════════════════════


class SQLAgentState(TypedDict):
    question: str
    sql_query: str
    sql_result: str
    sql_error: str
    final_answer: str
    retry_count: int


def sql_generate_sql(state: SQLAgentState) -> dict:
    """Node 1: LLM generates SQL from natural language question."""
    llm = get_llm()
    prompt = f"""You are a SQL expert. Given a question and the schema below, generate a SQLite SQL query to answer it.

Schema:
{TABLE_SCHEMA_PROMPT}

Rules:
- Output ONLY the SQL query, no explanations or markdown.
- Use only SELECT queries (read-only).
- Match brand/carline/model names exactly as stored.
- Use LIKE for fuzzy text matching.
- Limit results to 50 rows max.

Question: {state['question']}
SQL:"""
    response = llm.invoke(prompt)
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return {"sql_query": sql}


def sql_execute(state: SQLAgentState) -> dict:
    """Node 2: Execute the SQL query against SQLite."""
    result = run_sql(state["sql_query"])
    if result["success"]:
        return {"sql_result": result["data"], "sql_error": ""}
    return {"sql_result": "", "sql_error": result["error"]}


def sql_format_answer(state: SQLAgentState) -> dict:
    """Node 3: LLM formats the SQL result into a natural language answer."""
    llm = get_llm()
    prompt = f"""You are a data analyst. Explain these query results to a non-technical user.

Question: {state['question']}
SQL: {state['sql_query']}
Result: {state['sql_result']}

Provide a clear, concise summary of what was found. Highlight key parameter values.
Keep it under 3 paragraphs."""
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


def sql_should_retry(state: SQLAgentState) -> str:
    """Conditional edge: retry if SQL error and retry count < 2."""
    if state["sql_error"] and state["retry_count"] < 2:
        return "fix_sql"
    return "format_answer"


def sql_fix_sql(state: SQLAgentState) -> dict:
    """Fix node: LLM fixes the SQL query based on the error message."""
    llm = get_llm()
    prompt = f"""The SQL query below returned an error. Fix it.

Schema:
{TABLE_SCHEMA_PROMPT}

Original question: {state['question']}
Faulty SQL: {state['sql_query']}
Error: {state['sql_error']}

Output ONLY the corrected SQL query, no explanations."""
    response = llm.invoke(prompt)
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return {"sql_query": sql, "retry_count": state["retry_count"] + 1}


def build_sql_agent():
    """Build and compile the SQL Agent LangGraph."""
    from langgraph.graph import StateGraph, END

    builder = StateGraph(SQLAgentState)

    builder.add_node("generate_sql", sql_generate_sql)
    builder.add_node("execute_sql", sql_execute)
    builder.add_node("fix_sql", sql_fix_sql)
    builder.add_node("format_answer", sql_format_answer)

    builder.set_entry_point("generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_conditional_edges("execute_sql", sql_should_retry, {
        "fix_sql": "fix_sql",
        "format_answer": "format_answer",
    })
    builder.add_edge("fix_sql", "execute_sql")
    builder.add_edge("format_answer", END)

    return builder.compile()


@st.cache_resource(show_spinner="⚙️ Building SQL Agent...")
def get_sql_agent():
    return build_sql_agent()


# ═══════════════════════════════════════════════════════════════════════════
# RAG Agent — LangGraph Definition
# ═══════════════════════════════════════════════════════════════════════════


class RAGAgentState(TypedDict):
    question: str
    contexts: List[str]
    final_answer: str


def rag_retrieve(state: RAGAgentState) -> dict:
    """Node 1: Semantic search via ChromaDB."""
    from langchain_community.vectorstores import Chroma
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class LocalEmb:
        '''ChromaDB ONNX embedding (no API key needed).'''
        def __init__(self):
            self._ef = ONNXMiniLM_L6_V2()
        def embed_query(self, text):
            return [float(v) for v in self._ef([text])[0]]
        def embed_documents(self, texts):
            return [[float(v) for v in vec] for vec in self._ef(texts)]

    # Build vector store from SQLite data
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM vehicle_parameters")
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    conn.close()

    docs, metas, ids = [], [], []
    for row in rows:
        r = dict(zip(cols, row))
        text = f"Brand: {r['brand']} | Model: {r['model']} | System: {r['system']} | Param: {r['param_name']} | Group: {r['param_group']} | Type: {r['data_type']} | Desc: {r['description']}"
        docs.append(text)
        metas.append({'brand': r['brand'], 'param': r['param_name'], 'group': r['param_group']})
        ids.append(str(r['id']))

    vs = Chroma.from_texts(docs, LocalEmb(), metadatas=metas, ids=ids, collection_name="vehicle_params")
    results = vs.similarity_search(state["question"], k=5)
    return {"contexts": [d.page_content for d in results]}


def rag_generate(state: RAGAgentState) -> dict:
    """Node 2: Try LLM first, fall back to keyword matching."""
    contexts = state["contexts"]
    
    # Try LLM
    try:
        llm = get_llm()
        context_block = "\n".join(contexts)
        prompt = f"""You are a vehicle parameter expert. Below is the complete database.

Database:
{context_block}

Question: {state["question"]}

Answer using ONLY the database above. Refer to specific parameter names and values."""
        response = llm.invoke(prompt)
        return {"final_answer": response.content}
    except Exception:
        pass  # fall through to keyword matching

    # Fallback: keyword matching
    question = state["question"].lower()
    keywords = set(question.split())
    matched = [doc for doc in contexts if any(k in doc.lower() for k in keywords)]

    if matched:
        result = "Based on the database:\n"
        for doc in matched:
            result += "- " + doc + "\n"
    else:
        result = "Complete database:\n"
        for doc in contexts:
            result += "- " + doc + "\n"

    return {"final_answer": result}



def build_rag_agent():
    from langgraph.graph import StateGraph, END

    builder = StateGraph(RAGAgentState)

    builder.add_node("retrieve", rag_retrieve)
    builder.add_node("generate", rag_generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


@st.cache_resource(show_spinner="⚙️ Building RAG Agent...")
def get_rag_agent():
    return build_rag_agent()


# ═══════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════


with st.sidebar:
    st.markdown("## 🤖 AI Agent Hub")
    st.markdown("---")

    mode = st.radio(
        "Agent Mode",
        ["SQL Agent", "RAG Agent"],
        help="SQL Agent queries the database directly. RAG Agent searches semantic knowledge via ChromaDB.",
    )

    st.markdown("---")
    st.markdown("### 💡 Sample Queries")
    for i, q in enumerate(SAMPLE_QUERIES[mode]):
        if st.button(q, use_container_width=True, key=f"sample_{mode}_{i}"):
            st.session_state.pending_query = q

    st.markdown("---")
    st.markdown("### ⚙️ Status")
    has_key = LLM_API_KEY and LLM_API_KEY != "sk-your-api-key-here"
    if has_key:
        st.success("✅ LLM configured")
    else:
        st.error("❌ LLM not configured")

    st.markdown("### 📁 Data")
    conn = sqlite3.connect(DB_PATH)
    row_count = conn.execute("SELECT COUNT(*) FROM vehicle_parameters").fetchone()[0]
    brand_count = conn.execute("SELECT COUNT(DISTINCT brand) FROM vehicle_parameters").fetchone()[0]
    conn.close()
    st.markdown(f"- **{row_count}** parameters")
    st.markdown(f"- **{brand_count}** brands")

    st.markdown("---")
    st.markdown("<small>LangGraph + LangChain + Streamlit</small>", unsafe_allow_html=True)

# ── Main Area ─────────────────────────────────────────────────────────────

st.title(f"🔍 {mode}")

if mode == "SQL Agent":
    st.markdown(
        "Ask questions about vehicle parameters in natural language. "
        "The agent generates SQL, queries the database, and explains the results."
    )
else:
    st.markdown(
        "Ask questions about vehicle parameters using semantic search. "
        "The agent retrieves relevant knowledge from ChromaDB and generates an answer."
    )

# ── Chat History ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg and msg["details"]:
            with st.expander("🔍 Show details"):
                st.code(msg["details"], language="text")

# ── Chat Input & Processing ───────────────────────────────────────────────

if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    prompt = st.chat_input("Ask a question about vehicle parameters...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "details": ""})

    if not has_key:
        with st.chat_message("assistant"):
            st.error(
                "⚠️ **API Key not configured.**\n\n"
                "Set your `OPENAI_API_KEY` in `.env` or Streamlit Cloud Secrets."
            )
        st.session_state.messages.append({
            "role": "assistant", "content": "⚠️ API Key not configured.", "details": "",
        })
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                if mode == "SQL Agent":
                    agent = get_sql_agent()
                    result = agent.invoke({
                        "question": prompt,
                        "sql_query": "",
                        "sql_result": "",
                        "sql_error": "",
                        "final_answer": "",
                        "retry_count": 0,
                    })
                    answer = result["final_answer"]
                    details = f"SQL: {result['sql_query']}\n\nResult:\n{result.get('sql_result', '(empty)')}"
                else:
                    agent = get_rag_agent()
                    result = agent.invoke({
                        "question": prompt,
                        "contexts": [],
                        "final_answer": "",
                    })
                    answer = result["final_answer"]
                    details = "Retrieved Context:\n\n" + "\n\n".join(
                        f"--- Doc {i+1} ---\n{doc}" for i, doc in enumerate(result["contexts"])
                    )

                st.markdown(answer)
                with st.expander("🔍 Show details"):
                    st.code(details, language="text")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "details": details,
                })

            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg, "details": "",
                })
