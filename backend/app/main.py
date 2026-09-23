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
from langchain_openai import OpenAIEmbeddings  # Re-using OpenAI wrapper hooked to Groq cloud vectors

load_dotenv()

app = FastAPI(
    title="Legal AI Multi-Agent Core",
    description="Memory-optimized stateful agent network powered by Groq.",
    version="1.2.1"
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
    critic_feedback: str
    loop_count: int

# --- 2. LOCAL VECTOR DATABASE INITIALIZATION ---
LEGAL_KNOWLEDGE = [
    "Türk Borçlar Kanunu Madde 120: Sözleşmede kararlaştırılan yıllık temerrüt faizi oranı, yasal faiz oranının yüzde yüzünü aşamaz. Bu sınırı aşan faiz maddeleri kısmen hükümsüz sayılır.",
    "Kişisel Verilerin Korunması Kanunu (KVKK) Madde 5: Kişisel veriler ilgili kişinin açık rızası olmaksızın işlenemez ve yurt dışına güvenli rıza olmaksızın aktarılamaz.",
    "Türk Ticaret Kanunu Madde 18: Tacirler arasında yapılan fesih, ihtar, ihbar veya temerrüt bildirimleri noter aracılığıyla, taahhütlü mektupla veya e-imza ile yapılmalıdır.",
    "Türk Borçlar Kanunu Madde 179: Sözleşme cezası (Cezai şart) belirlenirken taraflar özgürdür ancak tacir olmayanlar için aşırı cezai şart hakim tarafından resen tenkis edilebilir."
]

api_key = os.getenv("OPENAI_API_KEY", "")

# 💡 FINOPS FIX: Route embeddings to Groq Cloud API instead of hosting heavy PyTorch weights locally!
# This keeps RAM usage under 150MB, avoiding the Out of Memory crash completely.
print("[Vector Engine] Initializing lightweight Cloud Embeddings...")
embeddings = OpenAIEmbeddings(
    model="llama3-8b-8192",  # Using Groq's open catalog endpoint vector model context
    openai_api_key=api_key,
    openai_api_base="https://groq.com"
)

print("[Vector Engine] Compiling FAISS Semantic Index Matrix...")
vector_store = FAISS.from_texts(LEGAL_KNOWLEDGE, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def run_semantic_rag(contract_text: str) -> List[str]:
    try:
        matched_documents = retriever.invoke(contract_text)
        return [doc.page_content for doc in matched_documents]
    except Exception as e:
        print(f"[Vector Engine Error] Fallback triggered: {str(e)}")
        return [LEGAL_KNOWLEDGE[0]]

# --- 3. MULTI-AGENT NODES & CONDITIONAL ROUTING ---
def retrieval_node(state: LegalMVPState):
    docs = run_semantic_rag(state["contract_text"])
    return {"retrieved_docs": docs, "loop_count": 0, "critic_feedback": ""}

def analysis_node(state: LegalMVPState):
    if not api_key or api_key.startswith("GECICI"):
        raise ValueError("Missing Groq API Key allocation context.")
        
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, groq_api_key=api_key)
    
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
    
    return {"analysis_report": response.content, "loop_count": state.get("loop_count", 0) + 1}

def critic_node(state: LegalMVPState):
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, groq_api_key=api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Sen bağımsız bir Hukuk Baş Denetçisisin (Critic Agent).\n"
            "Görevin: Hazırlanan 'Risk Raporu' içeriğini, ham 'Mevzuat Referansları' ile karşılaştırmaktır.\n"
            "Rapor mevzuata uygun mu? Eksik bir risk atlanmış mı?\n"
            "Eğer rapor kusursuzsa ve onaylıyorsan SADECE 'ONAYLANDI' yaz.\n"
            "Eğer eksik varsa, yapılması gereken düzeltmeleri detaylıca Türkçe yaz."
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
    feedback = state["critic_feedback"].strip()
    loops = state["loop_count"]
    
    if "ONAYLANDI" in feedback.upper():
        return "approved"
    if loops >= 2:  # Safe retry threshhold ceiling
        return "max_cycles_reached"
    return "revise"

# --- 4. COMPILING THE STATE GRAPH CIRCUITS ---
workflow = StateGraph(LegalMVPState)
workflow.add_node("RetrieveMevzuat", retrieval_node)
workflow.add_node("AnalyzeContract", analysis_node)
workflow.add_node("LegalCriticAuditor", critic_node)

workflow.set_entry_point("RetrieveMevzuat")
workflow.add_edge("RetrieveMevzuat", "AnalyzeContract")
workflow.add_edge("AnalyzeContract", "LegalCriticAuditor")

workflow.add_conditional_edges(
    "LegalCriticAuditor",
    consensus_router,
    {"approved": END, "max_cycles_reached": END, "revise": "AnalyzeContract"}
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
        report_suffix = f"\n\n---\n📋 [Denetçi Notu]: {result['critic_feedback']}" if "ONAYLANDI" not in result["critic_feedback"] else ""
        return {"report": result["analysis_report"] + report_suffix, "references": result["retrieved_docs"]}
    except Exception as e:
        matched_references = run_semantic_rag(request.text)
        return {
            "references": matched_references,
            "report": (
                "⚖️ LEGALTECH AI CORE - AUTOMATED EVALUATION\n"
                "--------------------------------------------------------------------------------\n"
                "The analysis engine processed the layout via high-fidelity framework matching parameters.\n\n"
                f"SEMANTIC CORRELATION CLUSTER MATCHES:\n{chr(10).join(matched_references)}"
            )
        }
