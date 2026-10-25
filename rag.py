import os
import re
import tempfile
from typing import List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------
# Page Setup
# ----------------------------------------------------------
st.set_page_config(
    page_title="DocuMind RAG • 100% Tailored CV & Job Match Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# Advanced CSS Animations & Workspace Styling
# ----------------------------------------------------------
st.markdown(
    """
<style>
    .stApp {
        background-color: #080c14;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', 'Fira Code', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    @keyframes workspaceUnlock {
        0% {
            opacity: 0;
            transform: scale(0.97) translateY(20px);
            filter: blur(8px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
            filter: blur(0);
        }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 12px rgba(56, 189, 248, 0.2); border-color: #1e293b; }
        50% { box-shadow: 0 0 28px rgba(56, 189, 248, 0.5); border-color: #38bdf8; }
        100% { box-shadow: 0 0 12px rgba(56, 189, 248, 0.2); border-color: #1e293b; }
    }

    @keyframes radarBlink {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .unlocked-view {
        animation: workspaceUnlock 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    .animated-header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        animation: pulseGlow 6s infinite ease-in-out;
    }

    .cyber-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 5s ease infinite;
        letter-spacing: -0.5px;
    }

    .radar-dot {
        width: 10px;
        height: 10px;
        background-color: #4ade80;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: radarBlink 1.8s infinite;
    }

    .status-badge-animated {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.35);
        margin-top: 10px;
    }

    div[data-testid="stChatMessage"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
    }

    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: #ffffff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.25s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
        border-color: #818cf8;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Sidebar: Settings & Configuration
# ----------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="API Key درج کریں تاکہ سسٹم ان لاک ہو سکے۔",
    )

    st.markdown("---")
    st.markdown("### 🎯 Workflow Mode")
    analysis_mode = st.radio(
        "Select Mode", ["📄 Document Q&A", "🎯 CV / Resume Tailoring & Rebuilder"], index=1
    )

    st.markdown("---")
    st.markdown("### 🤖 Neural Engine")
    model_name = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0
    )

    with st.expander("🔬 Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size", 400, 2500, 1200, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 400, 200, step=50)
        top_k = st.slider("Retrieved Chunks (k)", 1, 15, 8)

    st.markdown("---")
    if st.button("🔄 Clear Workspace", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------
# Security Lock Screen Check
# ----------------------------------------------------------
if not api_key:
    st.markdown(
        """
    <div style="text-align: center; padding: 70px 20px; border: 2px dashed #1e293b; border-radius: 16px; background: rgba(15, 23, 42, 0.4);">
        <div style="font-size: 54px; margin-bottom: 12px;">🔒</div>
        <h2 style="color: #38bdf8; margin-bottom: 8px;">Neural Workspace Locked</h2>
        <p style="color: #94a3b8; font-size: 15px; max-width: 500px; margin: 0 auto 18px auto;">
            ورک اسپیس کو چالو کرنے کے لیے سائڈبار میں اپنی <b>OpenAI API Key</b> درج کریں۔ تمام فیچرز، ملٹی فارمیٹ فائل لوڈر اور JD-مفصل ریزیومے انجن فوری ان لاک ہو جائیں گے۔
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ----------------------------------------------------------
# Unlocked Animated Workspace View
# ----------------------------------------------------------
st.markdown('<div class="unlocked-view">', unsafe_allow_html=True)

st.markdown(
    """
<div class="animated-header-card">
    <div class="cyber-title">⚡ DocuMind RAG & JD-Tailored CV Architect</div>
    <div style="color: #94a3b8; font-size: 14px; margin-top: 4px;">
        100% Job Description Alignment • Target Keyword Extraction • Gap Fulfillment • Complete ATS Rebuild
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Multi-Format Universal Document Loader
# ----------------------------------------------------------
def load_any_document(uploaded_file) -> List[Document]:
    filename = uploaded_file.name.lower()
    suffix = "." + filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if filename.endswith(".pdf"):
            return PyPDFLoader(tmp_path).load()
        elif filename.endswith(".docx"):
            return Docx2txtLoader(tmp_path).load()
        elif filename.endswith(".txt"):
            return TextLoader(tmp_path, encoding="utf-8").load()
        elif filename.endswith(".csv"):
            return CSVLoader(tmp_path, encoding="utf-8").load()
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    finally:
        os.unlink(tmp_path)


def build_vectorstore(uploaded_file, size: int, overlap: int) -> FAISS:
    docs = load_any_document(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------------------------------------
# Plotly Radial Gauge Chart
# ----------------------------------------------------------
def render_score_gauge(score: int):
    if score >= 80:
        bar_color = "#10b981"
    elif score >= 60:
        bar_color = "#38bdf8"
    elif score >= 40:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 42, "color": "#f8fafc"}},
            title={
                "text": "ORIGINAL ATS & JD FIT MATCH",
                "font": {"size": 13, "color": "#94a3b8"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#334155",
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#0f172a",
                "borderwidth": 1,
                "bordercolor": "#1e293b",
                "steps": [
                    {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                    {"range": [40, 75], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#e2e8f0", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        height=240,
        margin=dict(l=25, r=25, t=35, b=10),
    )
    return fig


# ----------------------------------------------------------
# Document Upload & Job Description Intake
# ----------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "📂 Upload Candidate CV/File (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
    )

if uploaded_file is not None:
    file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)
    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("⚡ Parsing document & generating vector embeddings..."):
            st.session_state.vectorstore = build_vectorstore(
                uploaded_file, chunk_size, chunk_overlap
            )
            st.session_state.file_sig = file_sig
            st.session_state.messages = []
            st.session_state.doc_name = uploaded_file.name

    with col2:
        st.markdown(
            f"""
        <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; margin-top: 26px;">
            <div style="color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Target File</div>
            <div style="color: #38bdf8; font-weight: 700; font-size: 14px; margin: 4px 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">📄 {uploaded_file.name}</div>
            <span class="status-badge-animated"><span class="radar-dot"></span>{analysis_mode.split()[1]} Active</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Target Job Description Input Container
target_jd = ""
if analysis_mode == "🎯 CV / Resume Tailoring & Rebuilder":
    with st.expander("🎯 Target Job Description (Paste Here to Tailor 100%)", expanded=True):
        target_jd = st.text_area(
            "Paste Job Description / Requirements / Required Skills:",
            height=160,
            placeholder="e.g. We are looking for a Senior AI/ML Engineer with strong skills in Python, LangChain, FAISS, Docker, and Microservices architecture. Must have experience optimizing LLM prompts, latency, and building RAG pipelines...",
            help="پوسٹنگ کا پورا ٹیکسٹ یا ریکوائرمنٹس یہاں پیسٹ کریں تاکہ ریزیومے کو ہو بہو جاب کے مطابق 100% ڈھالا جا سکے۔",
        )

# ----------------------------------------------------------
# Prompts: RAG vs. JD-Tailored 100% CV Builder
# ----------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a precision AI technical assistant.
Rules:
1. Answer strictly using ONLY the provided context.
2. If the context lacks the answer, say: "I couldn't find that in the document."
3. Cite page numbers or sections accurately.

Context:
{context}"""

CV_AUDIT_SYSTEM_PROMPT = """You are an elite Tech Career Strategist, Lead Recruiter, and ATS Resume Architect.
Your task is to analyze the candidate's CV against the provided Target Job Description (JD) and produce:
1. A rigorous diagnostic fit analysis and original ATS compatibility score.
2. A complete, 100% re-engineered, interview-ready ATS resume tailored directly to the target JD.

Target Job Description Provided:
\"\"\"{job_description}\"\"\"

Respond in this exact markdown structure:

### 📊 1. Original CV Health & JD Fit Score (موجودہ حالت اور جاب سے مطابقت)
- **Overall CV Score: [Insert Number]%**
- **Breakdown:**
  * JD Keyword & Skills Overlap: [X]%
  * Experience Relevance & Impact (XYZ Formula): [X]%
  * Structure & ATS Parsing Integrity: [X]%
  * Grammar, Professional Tone & Action Verbs: [X]%

### ⚠️ 2. Identified Errors & Deficiencies (خامیاں اور غلطیاں)
- Formatting inconsistencies, passive voice, weak bullet points, grammatical bugs, or missing metrics.

### 🧩 3. Missing Critical JD Elements & Skills Gap (کن چیزوں کی کمی ہے)
- Specific tools, frameworks, concepts, or qualifications required by the Job Description that were absent or understated in the candidate's original CV.

---

### 🚀 4. 100% Tailored & Optimized Resume (مکمل طور پر جاب کے مطابق تیار شدہ سی وی)
Rewrite the user's entire resume cleanly below in clean markdown.
Rules for Rebuilding:
- **Professional Summary**: Inject the target job title and highlight key competencies specified in the JD.
- **Skills Section**: Structure into categories (Languages, Frameworks, Tools, Cloud/DevOps) highlighting the exact keywords from the JD.
- **Work Experience & Projects**: Convert every point to the Google XYZ formula ("Accomplished [X], as measured by [Y], by doing [Z]"). Seamlessly align past responsibilities with the day-to-day requirements of the target role without fabricating credentials.
- **Flawless Formatting**: Standard ATS headings, clear dates, zero typos.

[START REBUILT RESUME]
(Provide the complete, ready-to-copy, 100% tailored CV here)
[END REBUILT RESUME]

Context from CV:
{context}"""


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source Chunk - Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


def get_chain(vectorstore: FAISS, model: str, k: int, mode: str, jd_text: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0.1, streaming=True)

    if "CV" in mode:
        system_content = CV_AUDIT_SYSTEM_PROMPT.format(
            job_description=jd_text.strip() if jd_text.strip() else "No specific Job Description provided. Optimize for broad industry excellence and ATS standards.",
            context="{context}"
        )
    else:
        system_content = RAG_SYSTEM_PROMPT

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_content),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ----------------------------------------------------------
# Interactive Chat & One-Click JD Tailoring Execution
# ----------------------------------------------------------
if "vectorstore" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # One-Click Tailored Audit & Rebuilder Trigger
    if analysis_mode == "🎯 CV / Resume Tailoring & Rebuilder":
        col_btn1, col_btn2 = st.columns([1.5, 3])
        with col_btn1:
            button_label = (
                "⚡ Audit & Tailor 100% to JD"
                if target_jd.strip()
                else "⚡ Audit & Rebuild 100% (Generic)"
            )
            trigger_audit = st.button(button_label, use_container_width=True)
    else:
        trigger_audit = False

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    user_input = st.chat_input(
        "Ask questions about the document or CV adjustments..."
        if analysis_mode == "🎯 CV / Resume Tailoring & Rebuilder"
        else "Query the document vector space..."
    )

    prompt_to_run = None
    if trigger_audit:
        if target_jd.strip():
            prompt_to_run = (
                "Perform a thorough ATS audit comparing this CV against the Target Job Description. "
                "Calculate the initial percentage fit, identify missing keywords/metrics, and generate a 100% tailored, "
                "ATS-compliant rebuilt resume explicitly mapped to this job description."
            )
        else:
            prompt_to_run = (
                "Calculate the overall percentage of this CV, pinpoint all grammar/format flaws, identify missing sections, "
                "and rewrite the entire CV into a 100% flawless, fully optimized ATS resume."
            )
    elif user_input:
        prompt_to_run = user_input

    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        chain, retriever = get_chain(
            st.session_state.vectorstore,
            model_name,
            top_k,
            analysis_mode,
            target_jd,
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing semantic overlap with Job Description and rebuilding resume..."):
                retrieved_docs = retriever.invoke(prompt_to_run)

            # Live Typing Stream
            answer = st.write_stream(chain.stream(prompt_to_run))

            # Visual Gauge & 100% Download Handler
            if analysis_mode == "🎯 CV / Resume Tailoring & Rebuilder":
                score_match = re.search(
                    r"Overall CV Score:\s*(\d{1,3})%", answer, re.IGNORECASE
                )
                if score_match:
                    score_val = int(score_match.group(1))
                    score_val = max(0, min(100, score_val))

                    st.markdown("---")
                    st.plotly_chart(
                        render_score_gauge(score_val),
                        use_container_width=True,
                    )

                # Extract rebuilt resume for direct download
                rebuilt_match = re.search(
                    r"\[START REBUILT RESUME\](.*?)\[END REBUILT RESUME\]",
                    answer,
                    re.DOTALL,
                )
                if rebuilt_match:
                    rebuilt_text = rebuilt_match.group(1).strip()
                    st.download_button(
                        label="📥 Download 100% JD-Tailored Resume (.MD)",
                        data=rebuilt_text,
                        file_name="100_Percent_JD_Tailored_Resume.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            with st.expander("🛰️ Source Evidence & Chunks"):
                for idx, doc in enumerate(retrieved_docs, start=1):
                    page = doc.metadata.get("page", 0)
                    st.markdown(
                        f"**Chunk #{idx} — Section/Page {page + 1 if isinstance(page, int) else page}**"
                    )
                    st.code(
                        doc.page_content[:600]
                        + ("..." if len(doc.page_content) > 600 else ""),
                        language="markdown",
                    )

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info(
        "💡 اوپر کسی بھی فارمیٹ (PDF, DOCX, TXT, CSV) کی فائل اپلوڈ کریں تاکہ ماڈل اس کا جائزہ لے سکے۔"
    )

st.markdown("</div>", unsafe_allow_html=True)
