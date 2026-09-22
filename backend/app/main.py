import os
import sys

# Bulunduğumuz main.py'nin iki üst klasörünü (yani backend dizinini) sys.path'e zorla ekliyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(current_dir, ".."))

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.workflow import legal_mvp_graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Legal AI MVP Backend",
    description="Azure Container Apps üzerinde koşan üretim seviyesi LegalTech API altyapısı.",
    version="1.0.0"
)

# Azure Static Web Apps yüklendiğinde bize vereceği canlı URL'i buraya bağlayacağız.
# Lokal testler için varsayılan olarak http://localhost:3000 açık kalır.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str

# Azure Container Apps'in sunucunun sağlıklı çalışıp çalışmadığını denetlemesi (Health Check) için gerekli endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": os.getenv("AZURE_ENV", "production")}

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    initial_state = {
        "contract_text": request.text,
        "retrieved_docs": [],
        "analysis_report": ""
    }
    
    # LangGraph orkestrasyonunu tetikleme
    result = legal_mvp_graph.invoke(initial_state)
    
    return {
        "report": result["analysis_report"],
        "references": result["retrieved_docs"]
    }
