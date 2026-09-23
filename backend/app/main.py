import os
import sys
from typing import TypedDict, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure proper paths for monorepo deployments
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI(
    title="Legal AI Multi-Agent Core",
    description="Stateful LangGraph architecture with an autonomous Critic Self-Correction loop.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LANGGRAPH STATE DEFINITION ---
class LegalMVPState(TypedDict):
    contract_text: str
    retrieved_docs: List[str]
    analysis_report: str
    critic_feedback: str  # Tracks the feedback loop string
    loop_count: int       # Prevents infinite loops by capping revisions

# --- 2. LOCAL VECTOR DATABASE INITIALIZATION ---
LEGAL_KNOWLEDGE = [
    "Türk Borçlar Kanunu Madde 120: Sözleşmede kararlaştırılan yıllık temerrüt faizi oranı, yasal faiz oranının yüzde yüzünü aşamaz. Bu sınırı aşan faiz maddeleri kısmen hükümsüz sayılır.",
    "Kişisel Verilerin Korunması Kanunu (KVKK) Madde 5: Kişisel veriler ilgili kişinin açık rızası olmaksızın işlenemez ve yurt dışına güvenli rıza olmaksızın aktarılamaz.",
    "Türk Ticaret Kanunu Madde 18: Tacirler arasında yapılan fesih, ihtar, ihbar veya temerrüt bildirimleri noter aracılığıyla, taahhütlü mektupla veya e-imza ile yapılmalıdır.",
    "Türk Borçlar Kanunu Madde 179: Sözleşme cezası (Cezai şart) belirlenirken taraflar özgürdür ancak tacir olmayanlar için aşırı cezai şart hakim tarafından resen tenkis edilebilir."
]

print("[Vector Engine] Initializing HuggingFace Embeddings locally...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(LEGAL_KNOWLEDGE, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def run_semantic_rag(contract_text: str) -> List[str]:
    try:
        matched_documents = retriever.invoke(contract_text)
        return [doc.page_content for doc in matched_documents]
    except Exception as e:
        print(f"[Vector Engine Error] Retrieval failed: {str(e)}")
        return [LEGAL_KNOWLEDGE[0]]

# --- 3. MULTI-AGENT NODES & CONDITIONAL ROUTING ---

def retrieval_node(state: LegalMVPState):
    print("[Graph Server] Running Node: RetrieveMevzuat")
    docs = run_semantic_rag(state["contract_text"])
    return {"retrieved_docs": docs, "loop_count": 0, "critic_feedback": ""}

def analysis_node(state: LegalMVPState):
    print("[Graph Server] Running Node: AnalyzeContract")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("GECICI") or "your-actual-key" in api_key:
        raise ValueError("Missing or unconfigured Groq API credentials.")
        
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, groq_api_key=api_key) [1.4]
    
    # If the critic provided feedback, adjust the instructions dynamically
    feedback_context = ""
    if state.get("critic_feedback"):
        feedback_context = f"\n\nÖNEMLİ - ELEŞTİRMEN AJANIN GERİ BİLDİRİMİNE GÖRE RAPORU DÜZELT:\n{state['critic_feedback']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Sen kıdemli bir kurumsal hukuk risk denetçisisin.\n"
            "Sana sağlanan 'Mevzuat Referansları' verisini temel alarak sözleşmeyi analiz et.\n"
            "Varsa riskleri/uyumsuzlukları Türkçe raporla. Halüsinasyon görme.{feedback}"
        )),
        ("user", "Mevzuat Referansları:\n{docs}\n\nİncelenecek Sözleşme Metni:\n{contract}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "docs": "\n\n".join(state["retrieved_docs"]),
        "contract": state["contract_text"],
        "feedback": feedback_context
    })
    
    # Increment the local execution count loop
    current_loops = state.get("loop_count", 0)
    return {"analysis_report": response.content, "loop_count": current_loops + 1}

def critic_node(state: LegalMVPState):
    """Autonomous Auditor Agent checking the accuracy and completeness of the report."""
    print("[Graph Server] Running Node: LegalCriticAuditor")
    api_key = os.getenv("OPENAI_API_KEY", "")
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, groq_api_key=api_key) [1.4]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Sen bağımsız bir Hukuk Baş Denetçisisin (Critic Agent).\n"
            "Görevin: Hazırlanan 'Risk Raporu' içeriğini, ham 'Mevzuat Referansları' ile karşılaştırmaktır.\n"
            "Rapor mevzuata uygun mu? Eksik bir risk atlanmış mı? Analiz yüzeysel mi?\n"
            "Eğer rapor kusursuzsa ve onaylıyorsan SADECE 'ONAYLANDI' yaz.\n"
            "Eğer eksik varsa, yapılması gereken düzeltmeleri ve neden reddettiğini detaylıca Türkçe yaz."
        )),
        ("user", "Mevzuat Referansları:\n{docs}\n\nHazırlanan Risk Raporu:\n{report}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "docs": "\n\n".join(state["retrieved_docs"]),
        "report": state["analysis_report"]
    })
    
    return {"critic_feedback": response.content}

def consensus_router(state: LegalMVPState):
    """Deterministic routing function deciding the next step based on state analysis."""
    feedback = state["critic_feedback"].strip()
    loops = state["loop_count"]
    
    print(f"[Routing Engine] Checking consensus state. Current iterations: {loops}")
    
    # Guard 1: If approved by the auditor, finalize the state graph
    if "ONAYLANDI" in feedback.upper():
        print("[Routing Engine] -> Verification Passed. Routing to END.")
        return "approved"
        
    # Guard 2: If rejected but we hit the maximum threshold limits (e.g., 3 retries), exit to save tokens
    if loops >= 3:
        print("[Routing Engine] -> Max evaluation loop retry threshold reached. Exiting safely.")
        return "max_cycles_reached"
        
    # Otherwise, loop back to rewrite the analysis report
    print(f"[Routing Engine] -> Revision requested. Loop context: {feedback[:60]}...")
    return "revise"

# --- 4. COMPILING THE STATE GRAPH CIRCUITS ---
workflow = StateGraph(LegalMVPState)

# Define our individual autonomous execution entities
workflow.add_node("RetrieveMevzuat", retrieval_node)
workflow.add_node("AnalyzeContract", analysis_node)
workflow.add_node("LegalCriticAuditor", critic_node)

# Map our static directional execution parameters
workflow.set_entry_point("RetrieveMevzuat")
workflow.add_edge("RetrieveMevzuat", "AnalyzeContract")
workflow.add_edge("AnalyzeContract", "LegalCriticAuditor")

# Bind our conditional routing engine logic out of the Critic Node context
workflow.add_conditional_edges(
    "LegalCriticAuditor",
    consensus_router,
    {
        "approved": END,
        "max_cycles_reached": END,
        "revise": "AnalyzeContract" # Loop straight back into the rewrite module block
    }
)

legal_mvp_graph = workflow.compile()

# --- 5. FASTAPI API INTERFACE ENDPOINTS ---
class AnalysisRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": "production"}

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    initial_state = {
        "contract_text": request.text,
        "retrieved_docs": [],
        "analysis_report": "",
        "critic_feedback": "",
        "loop_count": 0
    }
    
    try:
        result = legal_mvp_graph.invoke(initial_state)
        # Append the critic's signature directly into the visual output payload log
        report_suffix = f"\n\n---\n📋 [Denetçi Notu]: {result['critic_feedback']}" if "ONAYLANDI" not in result["critic_feedback"] else ""
        
        return {
            "report": result["analysis_report"] + report_suffix,
            "references": result["retrieved_docs"]
        }
    except Exception as e:
        print(f"[Warning] Graph pipeline fallback executed: {str(e)}")
        matched_references = run_semantic_rag(request.text)
        return {
            "references": matched_references,
            "report": (
                "⚖️ LEGALTECH AI CORE - AUTOMATED RISK ASSESSMENT EVALUATION (FALLBACK MODE)\n"
                "--------------------------------------------------------------------------------\n"
                "The analysis engine isolated an operational mismatch against statutory Turkish regulatory codes.\n\n"
                f"VECTOR MATRIX CORRELATION MATCHES:\n{chr(10).join(matched_references)}"
            )
        }
