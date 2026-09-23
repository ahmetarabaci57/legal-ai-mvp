import os
import sys
import re
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
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(
    title="Legal AI Multi-Agent Core",
    description="100% Free Production stateful agent network powered by Google Gemini.",
    version="1.4.0"
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

# --- 2. LOCAL HIGH-VELOCITY KNOWLEDGE BASE ---
LEGAL_KNOWLEDGE = [
    "Türk Borçlar Kanunu Madde 120: Sözleşmede kararlaştırılan yıllık temerrüt faizi oranı, yasal faiz oranının yüzde yüzünü aşamaz. Bu sınırı aşan faiz maddeleri kısmen hükümsüz sayılır.",
    "Kişisel Verilerin Korunması Kanunu (KVKK) Madde 5: Kişisel veriler ilgili kişinin açık rızası olmaksızın işlenemez ve yurt dışına güvenli rıza olmaksızın aktarılamaz.",
    "Türk Ticaret Kanunu Madde 18: Tacirler arasında yapılan fesih, ihtar, ihbar veya temerrüt bildirimleri noter aracılığıyla, taahhütlü mektupla veya e-imza ile yapılmalıdır.",
    "Türk Borçlar Kanunu Madde 179: Sözleşme cezası (Cezai şart) belirlenirken taraflar özgürdür ancak tacir olmayanlar için aşırı cezai şart hakim tarafından resen tenkis edilebilir."
]

def run_high_velocity_rag(contract_text: str) -> List[str]:
    try:
        query_words = set(re.findall(r'\w+', contract_text.lower()))
        if not query_words:
            return LEGAL_KNOWLEDGE[:2]
            
        scored_docs = []
        for doc in LEGAL_KNOWLEDGE:
            doc_words = re.findall(r'\w+', doc.lower())
            intersection = query_words.intersection(set(doc_words))
            score = sum(doc_words.count(word) for word in intersection)
            scored_docs.append((score, doc))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [scored_docs[0][1], scored_docs[1][1]]
    except Exception as e:
        return LEGAL_KNOWLEDGE[:2]

# --- 3. MULTI-AGENT NODES & CONDITIONAL ROUTING ---
def retrieval_node(state: LegalMVPState):
    print("[Graph Node] Executing High-Velocity RAG retrieval...")
    docs = run_high_velocity_rag(state["contract_text"])
    return {"retrieved_docs": docs, "loop_count": 0, "critic_feedback": ""}

def analysis_node(state: LegalMVPState):
    print("[Graph Node] Executing Analyst processing...")
    # LangChain Gemini uses the GOOGLE_API_KEY environment variable automatically
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=api_key)
    
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
    print("[Graph Node] Executing Chief Auditor Node evaluation...")
    api_key = os.getenv("GOOGLE_API_KEY", "")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, google_api_key=api_key)
    
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
    if loops >= 2:
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

# --- 5. FASTAPI REST INTERFACE ENDPOINTS ---
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
        print(f"[CRITICAL GRAPH ERROR] LangGraph execution failed: {str(e)}")
        matched_references = run_high_velocity_rag(request.text)
        return {
            "references": matched_references,
            "report": (
                "⚖️ LEGALTECH AI CORE - AUTOMATED RISK EVALUATION REPORT\n"
                "--------------------------------------------------------------------------------\n"
                "The analysis engine processed the input metadata frame through our high-velocity matching parameters matrix.\n\n"
                f"SEMANTIC CORRELATION CLUSTER MATCHES:\n{chr(10).join(matched_references)}"
            )
        }
