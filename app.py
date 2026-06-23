import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro HR Help Desk",
    page_icon="💼",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    .chat-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
        margin: 2px;
    }
    .out-of-scope {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1 style="margin:0;font-size:1.6rem">💼 Zyro Dynamics HR Help Desk</h1>
    <p style="margin:0.4rem 0 0;opacity:0.85;font-size:0.95rem">
        Ask any question about company policies, leave, benefits, and more.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH = "/kaggle/input/competitions/niat-masterclass-rag-challenge/zyro-dynamics-hr-corpus/"

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base from HR documents…")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512,
    )
    return retriever, llm


# ── Prompts ───────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the information in the context below.
If the context does not contain enough information, say so clearly.
Be concise and professional. Cite the policy document when possible.

Context:
{context}

Question: {question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Is the following question related to HR policies,
workplace rules, employee benefits, leave, compensation, conduct, performance,
IT security, travel expenses, onboarding, or separation at a company?

Reply with ONLY: "in-scope" or "out-of-scope".

Question: {question}
""")

REFUSAL = (
    "I can only answer questions related to Zyro Dynamics HR policies and "
    "company guidelines. This question appears to be outside that scope. "
    "Please reach out to the appropriate team for help."
)


# ── Core functions ────────────────────────────────────────────────────────────
def format_docs(docs):
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        pg = doc.metadata.get("page", "?")
        parts.append(f"[Source: {src}, Page {pg}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask_bot(question, retriever, llm):
    # Guardrail: classify question scope
    oos_response = llm.invoke(OOS_PROMPT.format(question=question))
    classification = oos_response.content.strip().lower()
    if "out-of-scope" in classification:
        return {"answer": REFUSAL, "source_documents": [], "out_of_scope": True}

    # RAG pipeline
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "source_documents": docs,
        "out_of_scope": False,
    }


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Available Policy Docs")
    policy_docs = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct",
        "Performance Review Policy", "Compensation & Benefits",
        "IT & Data Security", "POSH Policy",
        "Onboarding & Separation", "Travel & Expense Policy",
    ]
    for doc in policy_docs:
        st.markdown(f"• {doc}")
    st.divider()
    st.markdown("**Example questions:**")
    st.markdown("- How many earned leave days do I get?")
    st.markdown("- What is the WFH policy?")
    st.markdown("- How do I report harassment?")
    st.markdown("- What's the notice period for resignation?")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Build pipeline ────────────────────────────────────────────────────────────
try:
    retriever, llm = build_pipeline()
except Exception as e:
    st.error(f"Failed to initialize the knowledge base: {e}")
    st.stop()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("out_of_scope"):
            st.markdown(
                f'<div class="out-of-scope">⚠️ {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Sources:**")
            for s in msg["sources"]:
                st.markdown(
                    f'<span class="source-badge">📄 {s}</span>',
                    unsafe_allow_html=True,
                )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an HR policy question…"):
    # Save & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policy documents…"):
            result = ask_bot(prompt, retriever, llm)

        answer = result["answer"]
        sources = []
        out_of_scope = result["out_of_scope"]

        if out_of_scope:
            st.markdown(
                f'<div class="out-of-scope">⚠️ {answer}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(answer)
            if result["source_documents"]:
                sources = sorted(set(
                    doc.metadata.get("source", "Unknown").split("/")[-1]
                    for doc in result["source_documents"]
                ))
                st.markdown("**Sources:**")
                for s in sources:
                    st.markdown(
                        f'<span class="source-badge">📄 {s}</span>',
                        unsafe_allow_html=True,
                    )

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "out_of_scope": out_of_scope,
    })
