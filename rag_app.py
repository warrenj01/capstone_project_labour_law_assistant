import streamlit as st
import os
import sys
import re # Added for accountant tool to parse numbers
from dotenv import load_dotenv

# --- IMPORTS (STABLE LCEL ARCHITECTURE) ---
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
        groq_api_key = st.secrets["GROQ_API_key"]
    if not groq_api_key:
        st.error("FATAL: GROQ_API_KEY not found.")
        return None, None, None

    # 2. Initialize LLM
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
            st.error(f"Database not found at {CHROMA_DB_PATH}. Please run Task 3.")
            return None, None, None
            
        db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        # Changed k from 4 to 6 to increase search scope as requested
        retriever = db.as_retriever(search_kwargs={"k": 6}) 
    except Exception as e:
        st.error(f"Vector DB Error: {e}")
        return None, None, None

    return llm, retriever, db

# --- AGENT TOOLS ---

def tool_legal_research(llm, retriever, query):
    """Tool 1: RAG System for answering legal questions."""
    docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    rag_template = (
        "You are a confident and empathetic Labour Law Counsellor specializing in Mauritian legislation. "
        "Your primary goal is to provide clear, actionable, and legally sound guidance, making the user feel confident and supported. "
        
        "**TONE INSTRUCTION:** Do NOT start your answer with phrases like 'Based on the context...', 'According to the documents...', or 'As per the provided information...'. Speak with authority and confidence, integrating the information naturally into your advice, as a human expert would."
        "Crucially, when discussing paid leaves (sick leave or annual leave), always remind the user that entitlement often requires meeting minimum service periods, "
        "such as completing 12 months of continuous service for full annual leave. Specifically, paid leaves are typically *not* granted if an employee has been "
        "absent for a continuous period of 6 months or more, as per the law. "
        
        "Your answers must be accurate and based ONLY on the provided context. "
        "Context Sources:\n"
        "- WRA 2019 (Statutory Law)\n"
        "- Case Law (Judicial Precedent)\n"
        "- Cabinet Decisions (Government Policy)\n\n"
        
        "CASE LAW RULE: Prioritize quoting statutory law (WRA 2019) and general legal rules. **If citing a judicial precedent, you MUST provide the Case Name, Year, and Court (e.g., *Singh v. Employer Ltd. [2023] SC*).** Avoid cluttering general answers with case citations otherwise. Refer to parties internally as 'the employee' or 'the company' instead of Plaintiff/Defendant to maintain the friendly persona."
        
        "STRICT CONTENT RULE: Your output MUST focus purely on legal rules, definitions, and procedures. "
        "DO NOT include or repeat any specific numerical salaries, overtime calculations, or final figures (like Gross/Net Salary) from the conversation history or retrieved context, even if the context contains them. Only discuss numbers if the question is explicitly about a legal time duration or monetary threshold.\n\n"
        "REASONING INSTRUCTION: Before answering, explicitly check any time durations mentioned by the user against legal thresholds (e.g., 2 years service = 24 months vs 12 months threshold). "
        "STATUTORY INTERPRETATION RULE: For any entitlement (leave, bonus, allowance), you MUST determine if the law specifies a minimum service period (threshold) or if the entitlement is accrued on a **pro-rata** (proportional) basis, and explicitly state which rule applies based on the user's circumstances.\n\n"
        "PROCEDURAL INSTRUCTION: If the user's question is about *where to report a case* or *what legal steps to take*, "
        "and the context only contains details from a specific, unrelated court case (e.g., 'Plaintiff' and 'Defendant'), "
        "you MUST ignore the case specific names/details and instead provide the general, correct procedural answer. "
        "The standard initial reporting body for labor disputes in Mauritius is the Labour Office or the Ministry of Labour.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    
    prompt = ChatPromptTemplate.from_template(rag_template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({"context": context_text, "question": query})
    return response, docs

def tool_letter_writer(llm, query):
    """Tool 2: Generates official formal letters."""
    prompt = ChatPromptTemplate.from_template(
        "You are a professional secretary. Draft a formal, official letter regarding: {topic}. "
        "Use standard business letter formatting (Sender, Recipient, Date, Subject). "
        "Keep the tone professional, polite, and legally sound."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": query}), []

def tool_accountant(llm, query, history):
    """Tool 3: Handles calculations."""
    # Append the last three turns of history to the query for context
    context_history = [f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}" for m in history[-6:]]
    full_query_with_context = "\n".join(context_history) + f"\n\nLatest User Request: {query}"

    prompt = ChatPromptTemplate.from_template(
        "You are an expert payroll accountant for Mauritius. Your ONLY output is the calculation, formatted cleanly with steps. "
        "Crucial Instruction: Use the entire conversation history to calculate the user's net pay for the MONTH, based on the monthly salary and overtime provided. DO NOT include any conversational preamble or concluding remarks.\n"
        "**STRICT DEDUCTION/ALLOWANCE RULE:** Income Tax is a DEDUCTION. Do NOT invent or add any allowance (like MRA) unless explicitly mentioned in the user's conversation history as an 'allowance' or 'additional pay'. If the user mentions a specific Income Tax amount (e.g., 'Rs 1000'), use that value directly in the deduction step (Step 6) and **DO NOT** add it to the Gross Salary (Step 5).\n"
        "STRICT CALCULATION RULE: The output must ONLY contain the steps necessary to calculate the requested MONTHLY NET SALARY. DO NOT include any steps for Yearly Income, End of Year Bonus (EOY), or any other unsolicited calculation.\n"
        "Rules for Calculations:\n"
        "1. Monthly Basic Hours: Use 173.33 hours as the standard for converting monthly salary to an hourly rate.\n"
        "2. Hourly Rate: Calculate (Monthly Salary / 173.33).\n"
        "3. Overtime Rate: 1.5 times the calculated Hourly Rate (Time-and-a-half).\n"
        "4. Overtime Pay: Calculate (Overtime Hours * Overtime Rate).\n"
        "5. Gross Salary: Calculate (Monthly Salary + Overtime Pay) plus any explicitly known allowances (like MRA) provided by the user.\n"
        "6. Deductions: Calculate Net Pay by subtracting **NPS (5%)** and **CSG (5%)** from the Gross Salary. Income Tax should be included only if provided by the user (e.g., if they say it's 0, use 0).\n"
        "Show your working step-by-step. Re-state the final calculated Net Salary clearly.\n\n"
        "Full Context for Calculation: \n{full_query_with_context}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"full_query_with_context": full_query_with_context}), []

# --- FIX: Replaced simple string matching router with LLM-based router ---
def agent_router(llm, user_input, history):
    """The Investigator/Router, LLM-based for contextual routing."""
    
    # ----------------------------------------------------------------------
    # CRITICAL FIX 1: Explicit Logic Override for immediate Accounting Follow-ups
    # ----------------------------------------------------------------------
    force_accounting = False
    lower_input = user_input.lower()
    
    # Check 1: Initial Calculation Request (Addressing the user's first failure point)
    # Using regex to find keywords like "calculate", "pay", "salary", or "Rs" followed by a number
    if re.search(r'(calculate|salary|pay|overtime|rs)\s+[\d,.]+', lower_input) or \
       ("calculate" in lower_input and any(keyword in lower_input for keyword in ["salary", "pay", "overtime"])):
        force_accounting = True # <--- LINE 150 (Hard-coded initial calculation routing)
    
    # Check 2: Multi-Turn Update (Addressing the user's second failure point)
    # This remains the same, ensuring updates immediately follow calculations.
    if history and history[-1].type == 'ai': 
        last_response_message = history[-1].content
        # If the last response was a calculation AND the new input contains a number, force ACCOUNTING
        if "Step 1: Calculate" in last_response_message and re.search(r'\d+', user_input):
            force_accounting = True # <--- LINE 156 (Hard-coded multi-turn update routing)
            
    if force_accounting:
        return "ACCOUNTING" # <--- LINE 159 (Returns immediately if calculation is detected)
    # ----------------------------------------------------------------------
    
    system_prompt = (
        "You are a smart AI Coordinator 'Orchestrator'. You manage three experts:\n"
        "1. LEGAL: For questions about laws (WRA), court cases, or government policy announcements (Cabinet).\n"
        "2. LETTER: For requests to write, draft, or create official letters/emails.\n"
        "3. ACCOUNTING: For requests involving calculations, salary math, or tax numbers.\n\n"
        "**STRICT LEGAL OVERRIDE RULE:** If the query contains any legal keywords (e.g., 'eligible', 'entitled', 'law', 'vacation', 'right') AND specific high-value numbers (e.g., 50000, 75000), you MUST assume the user is asking about a legal threshold related to income and output 'LEGAL'. Do NOT route to accounting just because a number is present.\n"
        
        "**STRICT MULTI-TURN RULE (IMMEDIATE RECALCULATION):** If the immediately preceding assistant message was a detailed calculation (ACCOUNTING), the current user query is **overwhelmingly likely** to be an update, clarification, or adjustment to that calculation (e.g., 'What if tax was 1000?'). In this scenario, you **MUST** output 'ACCOUNTING' unless the user clearly changes the subject (e.g., 'Now write me a letter'). YOU MUST ALSO REDO the calculation immediately and show result(s) with the amended parameter, without asking the user if they want an updated calculation.\n"
        
        "**ROUTING ERROR PENALTY:** If the previous tool was 'ACCOUNTING' and the current query contains a number (e.g., '1000'), and you fail to output 'ACCOUNTING', you will receive a severe routing error penalty. Therefore, prioritize 'ACCOUNTING' in such follow-up scenarios above all other decisions, including 'GREETING'. "
        
        "**ROUTING PREFERENCE:** If the previous assistant message was a tool output (LEGAL, LETTER, ACCOUNTING), and the current query is not a new topic, always try to route back to the last used tool or output 'INVESTIGATE'. **NEVER output 'GREETING' in response to a direct follow-up.**"
        
        "ROUTING INSTRUCTION: Analyze the current user query in the context of the entire conversation history. "
        "You must flawlessly detect context switching. If the query is a general follow-up, check the history first.\n"
        
        "Your Decision Logic:\n"
        "- IF the user asks a general legal question (e.g., 'Do I get compensation?', 'Is this legal?'), even without specific details, "
        "output 'LEGAL'.\n"
        "- ONLY output 'INVESTIGATE' if the user specifically asks for a CALCULATION (e.g., 'How much will I get?') but is missing the numbers (Salary/Years).\n"
        "- IF the request is clear, output the tool name: 'LEGAL', 'LETTER', or 'ACCOUNTING'.\n"
        "- IF the input is a true, simple greeting (e.g., 'Hi', 'Hello'), output 'GREETING'.\n"
        "- IF the input is clearly unrelated to law, letters, or accounting (e.g., 'What is the capital of France?'), output 'OUT_OF_SCOPE'.\n"
        "Output ONLY the single word."
    )
    
    # Prepare messages for the LLM router
    messages = [SystemMessage(content=system_prompt)] + history[-3:] + [HumanMessage(content=user_input)]
    
    # Use the LLM to decide the route
    response = llm.invoke(messages)
    decision = response.content.strip().upper()
    
    if "LEGAL" in decision: return "LEGAL"
    if "LETTER" in decision: return "LETTER"
    if "ACCOUNTING" in decision: return "ACCOUNTING"
    if "INVESTIGATE" in decision: return "INVESTIGATE"
    if "OUT_OF_SCOPE" in decision: return "OUT_OF_SCOPE"
    return "GREETING"


def agent_investigator(llm, user_input):
    """Generates a clarifying question."""
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for a Mauritian Labour Law App. The user asked: '{input}'. \n"
        "This request needs specific details to be processed by our Calculation or Letter tools.\n"
        "Rules for your response:\n"
        "1. Do NOT ask for the country (Assume Mauritius).\n"
        "2. Only ask for the specific missing numbers/details needed (e.g., Monthly Salary, Years of Service).\n"
        "3. Keep it polite, short, and direct."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input})

def agent_sentiment(llm, user_input):
    """
    Analyzes the emotional tone of the user.
    Returns: 'NEGATIVE' (Distress/Anger), 'NEUTRAL', or 'POSITIVE'.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of this legal query: '{input}'. \n"
        "If the user seems distressed, angry, victimized, or mentions harassment/firing, return 'NEGATIVE'.\n"
        "If the user is asking a standard informational question, return 'NEUTRAL'.\n"
        "Return ONLY the single word."
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": user_input}).strip().upper()
    return response

def agent_context_handler(history, action):
    """
    Handles GREETING, OUT_OF_SCOPE, and fallback responses using context.
    """
    # 1. Determine the last expert used for context
    last_expert = "Labour Law or Payroll"
    # Find the last actual expert response (not just the initial greeting)
    relevant_messages = [m for m in history if m['role'] == 'assistant' and not m['content'].startswith("Hello! I am here to help")]
    
    # Check the last message content for strong keywords
    if relevant_messages:
        content = relevant_messages[-1]['content']
        if "Legal Researcher" in content or "legal information" in content or "Context Sources" in content: 
            last_expert = "Legal Research"
        elif "Secretary Agent" in content or "Draft a formal" in content: 
            last_expert = "Drafting a Letter"
        elif "Accountant Agent" in content or "Rules for Calculations" in content or "Gross Salary" in content: 
            last_expert = "Payroll Calculation"
    
    # 2. Generate a contextual response
    if action == "OUT_OF_SCOPE":
        return f"I am a specialized tool focused on Mauritian Labour Law, payroll calculations, and official letter drafting. The topic you raised seems to be outside my scope. Would you like to continue our discussion on **{last_expert}**, or do you have a new question related to my expertise?"
    
    # GREETING/Vague response handling
    if action == "GREETING":
        return f"Hello again! I am ready to continue our discussion on **{last_expert}**. Do you need further clarification, or would you like to switch to a new legal, calculation, or letter-drafting task? Remember you can use the 'Clear Conversation' button to reset for a completely new topic."
    
    # Fallback response
    return "How can I help you with your next legal, calculation, or drafting task?"


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Mauritian - Labour Law AI Assistant", layout="wide", page_icon="⚖️")

# --- MODERN POPUPS (DIALOGS) ---
@st.dialog("About")
def show_about_popup():
    st.header("Jean Michel Nelson")
    st.caption("GenAI & Machine Learning Bootcamp 2025")
    st.write("This Capstone Project demonstrates the power of Multi-Agent RAG systems in making legal information accessible.")
    st.markdown("**Project Date:** Nov-Dec 2025")
    st.markdown("**Technology:** LangChain, Streamlit, ChromaDB, Llama 3")

@st.dialog("Legal Disclaimer")
def show_disclaimer_popup():
    st.error("⚠️ Important Legal Notice")
    st.write("This application is an AI-powered research tool, not a lawyer.")
    st.markdown("""
    1. **Informational Only:** The content generated is for educational purposes.
    2. **Not Legal Advice:** Do not rely on this tool for court cases or binding contracts.
    3. **Consult a Professional:** Always verify information with a qualified Mauritian Attorney.
    """)

# --- MODERN STYLING ---
st.markdown("""
<style>
    /* Modern Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }
    
    /* Ensure chat input is visible and clean */
    .stChatInput {
        border-top: 1px solid #e0e0e0;
    }

    .stMarkdownContainer{
        font-size:0.7rem;
    }

    html {
        font-size:14px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Labour Law Helper - Agent System")
    st.info("System Status: **Online**")
    
    st.markdown("### 🛠️ Active Agents")
    st.success("👨‍⚖️ Legal Researcher")
    st.success("✍️ Letter Secretary")
    st.success("🧮 Payroll Accountant")
    
    st.markdown("---")
    st.markdown("### ℹ️ Information")
    
    # Modern Popup Buttons
    if st.button("👤 About Author", use_container_width=True):
        show_about_popup()
        
    if st.button("⚖️ Disclaimer", use_container_width=True):
        show_disclaimer_popup()
        
    st.markdown("---")
    if st.button("Clear Conversation", type="primary", use_container_width=True):
        # Clear the cache associated with load_core_resources before rerunning
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("Mauritian - Labour Law AI Assistant")
st.caption("Multi-Agent System powered by Groq (Llama 3) & RAG")

llm, retriever, db = load_core_resources()

if llm and retriever:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am here to help you with questions related to labour laws. I can direct you to our Legal Researcher, Accountant, or Secretary. How can I assist you today?"}]

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Area (Pins to bottom automatically)
    if prompt := st.chat_input("Ex: 'I have been harassed at work', 'Calculate severance', or 'Draft a resignation letter'"):
        
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Agent Processing
        with st.spinner("Orchestrator is routing request..."):
            
            history_lc = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user": history_lc.append(HumanMessage(content=m["content"]))
                else: history_lc.append(AIMessage(content=m["content"]))

            # --- STEP 1: SENTIMENT ANALYSIS (New Feature) ---
            sentiment_prefix = ""
            with st.spinner("Analyzing context & sentiment..."):
                emotion = agent_sentiment(llm, prompt)
                if emotion == "NEGATIVE":
                    # Removed the "Empathy Mode Active" label, just kept the message
                    sentiment_prefix = "*I am truly sorry to hear about your situation. I will do my best to provide the relevant legal information to assist you.*\n\n"
            
            # --- STEP 2: ROUTING (Now Contextual) ---
            
                # FIX: Pass LLM and history for contextual routing
                action = agent_router(llm, prompt, history_lc)
            
            response_text = ""
            sources = []
            
            # --- STEP 3: EXECUTION ---
            if action == "INVESTIGATE":
                response_text = agent_investigator(llm, prompt)
            elif action == "LEGAL":
                st.toast("🔍 Routing to Legal Researcher...", icon="⚖️")
                with st.spinner("Searching Case Law & WRA..."):
                    raw_response, sources = tool_legal_research(llm, retriever, prompt)
                    response_text = raw_response
                    # NOTE: Sentiment prefix applied at end of step 4
            elif action == "LETTER":
                st.toast("✍️ Routing to Secretary Agent...", icon="📝")
                with st.spinner("Drafting official document..."):
                    raw_response, sources = tool_letter_writer(llm, prompt)
                    response_text = raw_response
                    # NOTE: Sentiment prefix applied at end of step 4
            elif action == "ACCOUNTING":
                st.toast("🧮 Routing to Accountant Agent...", icon="🔢")
                with st.spinner("Calculating figures..."):
                    # Pass the full history to the accounting tool for better context
                    raw_response, sources = tool_accountant(llm, prompt, st.session_state.messages) 
                    response_text = raw_response
                    # NOTE: We skip the sentiment_prefix here to maintain a clean calculation output.
            
            # --- Handle GREETING / OUT_OF_SCOPE contextually ---
            elif action in ["GREETING", "OUT_OF_SCOPE"]:
                 response_text = agent_context_handler(st.session_state.messages, action)
            else: 
                # Default initial greeting fallback
                response_text = "Hello! I am ready to help with legal questions, letter drafting, or payroll calculations."

            # --- STEP 4: ADD CLOSING SUGGESTION ---
            # Apply closing suggestion only if it wasn't a calculation (which should be pure output)
            # OR if it's a contextual response. This simplifies the logic.
            if action != "ACCOUNTING":
                response_text = sentiment_prefix + response_text
                closing_suggestion = "\n\n***\nI'm here to discuss this further. If you have any follow-up questions, need this information clarified, or want to explore other options (like drafting a letter or calculating figures), please let me know."
                response_text += closing_suggestion
            else: 
                # For pure calculation, just add the closing suggestion to the end of the calculation block.
                closing_suggestion = "\n\n***\nI'm here to discuss this further. If you have any follow-up questions, need this information clarified, or want to explore other options (like drafting a letter or calculating figures), please let me know."
                response_text += closing_suggestion
            
            st.markdown(response_text)
            
            if sources:
                with st.expander("📚 Referenced Legal Documents"):
                    for i, doc in enumerate(sources):
                        src = doc.metadata.get('source', 'Unknown')
                        st.markdown(f"**{i+1}. {src}**")
                        st.caption(f"\"{doc.page_content[:200]}...\"")

            st.session_state.messages.append({"role": "assistant", "content": response_text})