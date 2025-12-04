import streamlit as st
import os
import sys
from dotenv import load_dotenv

# --- IMPORTS (STABLE LCEL ARCHITECTURE) ---
# We use pure LCEL components to avoid 'ModuleNotFoundError' on broken chain paths
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Integration components
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq 

# --- CONFIGURATION ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Robustly handle the path to the chroma db
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db_index")
BERT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
GENERATIVE_LLM_MODEL = "llama-3.1-8b-instant" 

# --- INITIALIZATION ---
@st.cache_resource
def load_core_resources():
    """Initializes the LLM and Database connections once."""
    
    # 1. API Key Check
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        st.error("FATAL: GROQ_API_KEY not found. Please check your .env file.")
        return None, None, None

    # 2. Initialize LLM (The Brain)
    try:
        llm = ChatGroq(api_key=groq_api_key, model=GENERATIVE_LLM_MODEL, temperature=0.0)
    except Exception as e:
        st.error(f"Groq Init Error: {e}")
        return None, None, None

    # 3. Initialize Embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name=BERT_EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Embeddings Error: {e}")
        return None, None, None

    # 4. Initialize Vector DB
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            st.error(f"Database not found at {CHROMA_DB_PATH}. Please run Task 3 (Vectorization) first.")
            return None, None, None
            
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Vector DB Error: {e}")
        return None, None, None

    return llm, retriever, db

# --- AGENT TOOLS (The Experts) ---

def tool_legal_research(llm, retriever, query):
    """Tool 1: RAG System for answering legal questions (Pure LCEL)."""
    
    # 1. Explicit Retrieval (No 'RetrievalQA' chain needed to avoid import errors)
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Generation Prompt
    rag_template = (
        "You are an expert legal assistant for Mauritian Labour Law. "
        "Your answers must be accurate and based ONLY on the provided context. "
        "Context Sources:\n"
        "- WRA 2019 (Statutory Law)\n"
        "- Case Law (Judicial Precedent)\n"
        "- Cabinet Decisions (Government Policy)\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(rag_template)
    chain = prompt | llm | StrOutputParser()
    
    # 3. Execute
    response = chain.invoke({"context": context_text, "question": query})
    return response, docs

def tool_letter_writer(llm, topic):
    """Tool 2: Generates official formal letters."""
    prompt = ChatPromptTemplate.from_template(
        "You are a professional secretary. Draft a formal, official letter regarding: {topic}. "
        "Use standard business letter formatting (Sender, Recipient, Date, Subject). "
        "Keep the tone professional, polite, and legally sound."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": topic}), []

def tool_accountant(llm, query):
    """Tool 3: Handles calculations and compensation logic."""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert payroll accountant for Mauritius. Help the user with this calculation/query: {query}. \n"
        "Rules:\n"
        "1. End of Year (EOY) Bonus / 13th Month: Formula is (Total Earnings in Year / 12). "
        "If the employee worked part of the year, it is pro-rated: (Monthly Salary * Months Worked) / 12.\n"
        "2. Severance: Standard formula is (Years of Service * 0.5 * Monthly Salary) unless specified otherwise.\n"
        "Show your working step-by-step. Return ONLY the calculation and explanation."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}), []

def agent_router(llm, user_input, history):
    """
    The Investigator/Router: 
    Analyzes input to decide if we need more info (Investigate) or can proceed to a tool.
    """
    system_prompt = (
        "You are a smart AI Coordinator 'Orchestrator'. You manage three experts:\n"
        "1. LEGAL: For questions about laws (WRA), court cases, or government policy announcements (Cabinet).\n"
        "2. LETTER: For requests to write, draft, or create official letters/emails.\n"
        "3. ACCOUNTING: For requests involving calculations, salary math, or tax numbers.\n\n"
        "Your Decision Logic:\n"
        "- IF the user asks a general legal question (e.g., 'Do I get compensation?', 'Is this legal?'), even without specific details, "
        "output 'LEGAL'. The researcher can explain the general rules.\n"
        "- ONLY output 'INVESTIGATE' if the user specifically asks for a CALCULATION (e.g., 'How much will I get?') but is missing the numbers (Salary/Years).\n"
        "- IF the request is clear, output the tool name: 'LEGAL', 'LETTER', or 'ACCOUNTING'.\n"
        "- IF the input is just a greeting, output 'GREETING'.\n"
        "Output ONLY the single word."
    )
    
    # We pass the last few messages to give the router context
    messages = [SystemMessage(content=system_prompt)] + history[-3:] + [HumanMessage(content=user_input)]
    
    response = llm.invoke(messages)
    decision = response.content.strip().upper()
    
    # Cleaning up decision string for robustness
    if "LEGAL" in decision: return "LEGAL"
    if "LETTER" in decision: return "LETTER"
    if "ACCOUNTING" in decision: return "ACCOUNTING"
    if "INVESTIGATE" in decision: return "INVESTIGATE"
    return "GREETING"

def agent_investigator(llm, user_input):
    """Generates a clarifying question to ask the user."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for a Mauritian Labour Law App. The user asked: '{input}'. \n"
        "This request needs specific details to be processed by our Calculation or Letter tools.\n"
        "Rules for your response:\n"
        "1. Do NOT ask for the country (Assume Mauritius).\n"
        "2. Only ask for the specific missing numbers/details needed (e.g., Monthly Salary, Years of Service).\n"
        "3. Keep it polite, short, and direct."
        "no more than 2 questions at a time. you can ask for other details later if needed.\n"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input})

# --- STREAMLIT UI ---

st.set_page_config(page_title="Legal AI Agent", layout="wide", page_icon="⚖️")

# Custom CSS for a refreshed look
st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; padding: 10px; border: 1px solid #e0e0e0; }
    .stSidebar { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    div[data-testid="stToast"] { padding: 10px; background-color: #e6f3ff; border-left: 5px solid #0068c9; }
</style>
""", unsafe_allow_html=True)

# Sidebar with Agents
with st.sidebar:
    st.title("Labour Law Helper - Agent System")
    st.info("Orchestrator: **Active**")
    
    st.markdown("### Available Experts")
    st.markdown("👨‍⚖️ **Legal Researcher**\n*Checks WRA, Cases, Cabinet*")
    st.markdown("✍️ **Secretary**\n*Drafts formal letters*")
    st.markdown("🖩 **Accountant**\n*For calculations related to employment*")
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Mauritius - Labour Law AI Assistant")
st.caption("Multi-Agent System powered by **Groq (Llama 3)** & RAG")

# Load Resources
llm, retriever, db = load_core_resources()

if llm and retriever:
    # Initialize History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am here to help you with questions related to labour laws. I can direct you to our Legal Researcher, Accountant, or Secretary. How can I assist you today?"}]

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Ex: 'Draft a formal letter', 'Calculate severance for 5 years', or 'What is the law on overtime?'"):
        
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Agent Processing
        with st.chat_message("assistant"):
            
            # Prepare history for the router (convert dict to LangChain messages)
            history_lc = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user": history_lc.append(HumanMessage(content=m["content"]))
                else: history_lc.append(AIMessage(content=m["content"]))

            # --- STEP 1: ROUTING & INVESTIGATION ---
            with st.spinner("Orchestrator is analyzing request..."):
                action = agent_router(llm, prompt, history_lc)
            
            response_text = ""
            sources = []
            
            # --- STEP 2: EXECUTION ---
            if action == "INVESTIGATE":
                response_text = agent_investigator(llm, prompt)
                
            elif action == "LEGAL":
                st.toast("🔍 Routing to Legal Researcher...", icon="⚖️")
                with st.spinner("Searching Case Law & WRA..."):
                    response_text, sources = tool_legal_research(llm, retriever, prompt)
                    
            elif action == "LETTER":
                st.toast("✍️ Routing to Secretary Agent...", icon="📝")
                with st.spinner("Drafting official document..."):
                    response_text, sources = tool_letter_writer(llm, prompt)

            elif action == "ACCOUNTING":
                st.toast("🧮 Routing to Accountant Agent...", icon="🔢")
                with st.spinner("Calculating figures..."):
                    response_text, sources = tool_accountant(llm, prompt)
                    
            else: 
                response_text = "Hello! I am ready to help with legal questions, letter drafting, or payroll calculations."

            # --- STEP 3: DISPLAY ---
            st.markdown(response_text)
            
            if sources:
                with st.expander("📚 Referenced Legal Documents"):
                    for i, doc in enumerate(sources):
                        src = doc.metadata.get('source', 'Unknown')
                        st.markdown(f"**{i+1}. {src}**")
                        st.caption(f"\"{doc.page_content[:200]}...\"")

            st.session_state.messages.append({"role": "assistant", "content": response_text})