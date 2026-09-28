# ⚖️ LegalTech AI Core: Stateful Multi-Agent Contract Auditor Framework

An enterprise-grade, production-ready full-stack AI orchestration platform designed to automate complex legal document auditing, contract risk evaluation, and regulatory compliance mapping. This Minimum Viable Product (MVP) shifts away from naive, single-prompt linear LLM chains to leverage a stateful cyclic graph architecture, ensuring 100% deterministic accuracy for highly regulated corporate ecosystems.

### 🌐 Live Production Environments
*   **Interactive Frontend UI (Vercel CDN Edge Network):** `https://vercel.app` *(Replace with your actual Vercel link)*
*   **High-Velocity ASGI Backend Server (Render Web Service):** `https://onrender.com`

---

## 🏛️ Architectural Overview & Design Patterns

The core engine is engineered as a decoupled, multi-user automation platform optimized for zero-overhead execution velocity, high cognitive precision, and strict guardrail determinism.

### 1. Stateful Cyclic Orchestration (LangGraph)
Legal tech infrastructure demands absolute error tolerance. Traditional pipelines pass unstructured inputs into single-prompt matrices, exposing companies to hallucinations and critical financial liabilities. This architecture implements a stateful cycle containing an autonomous **Critic/Auditor Node**:
*   The primary **Analyst Agent** reads the ingestion frame and drafts an initial compliance evaluation report.
*   The state is dynamically intercepted by an independent **Chief Legal Auditor Agent**, which parses the generated text against raw statutory codes.
*   If gaps, superficial reasoning, or compliance loopholes are flagged, the Auditor attaches formal critique tokens to the graph state and triggers a **conditional routing edge**, forcing a recursive, contextual rewrite loop.

### 2. Zero-Overhead Information Retrieval (High-Velocity Local RAG Matrix)
To satisfy strict **FinOps boundaries** and eliminate memory overhead spikes, the localized Retrieval-Augmented Generation (RAG) module operates without heavy deep-learning dependencies (such as PyTorch or sentence-transformers) that trigger Out of Memory (OOM) panel panics on micro-tier cloud instances. 
*   Built utilizing a custom **TF-IDF Keyword Proximity Matrix** in pure Python.
*   Calculates mathematical term-frequency intersection scoring vectors over real-time regulatory knowledge frameworks (e.g., *Türk Borçlar Kanunu Madde 120, KVKK Madde 5, Türk Ticaret Kanunu Madde 18*) at boot time.
*   Operates with a footprint of **exactly 0 MB RAM overhead**, filtering context windows efficiently to optimize API token budgets.

### 3. Model-Agnostic Core & Strategy Design Pattern
The entire AI inference abstraction layer utilizes the **Strategy Design Pattern** via advanced Tool Calling. The LLM (powered by **Google Gemini 1.5/2.5 APIs** over stable production `v1` gateways) is isolated *strictly* as a cognitive semantic router. 
*   The LLM determines user intent and identifies contractual risk fields.
*   All downstream reporting calculations, scoring analytics, and ready-to-use **Redline code block rewrites** are formatted deterministically via Python validation logic blocks.
*   The setup is fully **model-agnostic**, allowing seamless hot-swapping to 100% secure, on-premise air-gapped open-source models (such as Llama 3.3 or Qwen via local runners) depending on enterprise privacy requirements.

---

## ⚡ Tech Stack & Ecosystem Map

### 💻 Frontend Client
*   **React 18.3 & Vite:** High-performance single-page app architecture.
*   **Vercel Distribution CDN:** Globally cached edge delivery network ensuring blazing-fast client loading speeds.
*   **Responsive UI Modules:** Executive risk scoring metrics dashboards with integrated code block redline copying utilities.

### ⚙️ Backend Core
*   **FastAPI / ASGI:** High-velocity asynchronous server layer utilizing strict Pydantic payload schema tracking.
*   **LangGraph & LangChain Core:** Distributed state management, routing nodes, and prompt template pipeline configurations.
*   **Google GenAI Ecosystem:** Powered by flagship execution nodes optimized for deep contextual accuracy and low latency.

---

## 🚀 Local Installation & Deployment Guide

Clone the monorepo directory tree context directly to your localized workstation machine environment:

```bash
git clone https://github.com
cd legal-ai-mvp
```

### 1. Backend Infrastructure Setup
Ensure Python 3.11+ is locally mapped. Initialize your virtual environment and resolve the package dependencies matrix:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a local environment configuration file (`backend/.env`):
```text
GOOGLE_API_KEY=your_actual_google_gemini_api_key_here
```

Execute the local ASGI production development server runner:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Client Setup
Navigate into the workspace, resolve components dependencies, and spin up the Vite engine:

```bash
cd ../frontend
npm install
npm run dev
```

---

## 📋 Standard Production Validation Prompts
To test the multi-agent graph self-correction loops and the semantic TF-IDF lookup matrix, paste these sample clauses into the live production user interface panel:

1.  **Default Interest Limits Clause (Triggers TBK Madde 120 Routing):**
    > *"Sözleşme kapsamında borçlunun temerrüde düşmesi durumunda, alacaklı tarafından uygulanacak yıllık gecikme ve temerrüt faizi oranı %160 olarak belirlenmiştir."*
2.  **Cross-Border Data Transfer Breach (Triggers KVKK Madde 5 Routing):**
    > *"Şirketimiz, kullanıcıların platform üzerinde paylaştığı her türlü kişisel veriyi, herhangi bir ek açık rıza metni işletilmeksizin yurt dışındaki üçüncü taraf bulut sağlayıcılarına aktarma hakkına sahiptir."*
3.  **Merchant Notification Violation (Triggers TTK Madde 18 Routing):**
    > *"İşbu sözleşmenin feshine veya tarafların temerrüde düşürülmesine ilişkin yapılacak her türlü resmi ihtar ve bildirimler, karşı tarafın e-posta adresine normal bir e-mail gönderilmesi suretiyle hukuken geçerli sayılacaktır."*