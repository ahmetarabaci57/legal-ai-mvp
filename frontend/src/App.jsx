import React, { useState } from 'react';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    
    const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    try {
      const response = await fetch(`${apiBaseUrl}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Analysis Error:", error);
      setResult({ report: "An error occurred connecting to the AI analysis engine. Ensure your API settings are correct.", references: [] });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '40px', fontFamily: 'system-ui, sans-serif', maxWidth: '1200px', margin: '0 auto' }}>
      <header style={{ marginBottom: '30px' }}>
        <h1 style={{ color: '#1e293b', margin: '0 0 8px 0' }}>⚖️ LegalTech AI Core</h1>
        <p style={{ color: '#64748b', margin: 0 }}>Founding Engineer MVP - Contract Risk Analyzer Framework</p>
      </header>
      
      <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 500px', backgroundColor: '#fff', padding: '24px', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}>
          <h3 style={{ color: '#334155', marginTop: 0 }}>Sözleşme Metni (Contract Input)</h3>
          <textarea
            rows="14"
            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '14px', boxSizing: 'border-box', resize: 'vertical' }}
            placeholder="Analiz edilecek sözleşme maddelerini buraya yapıştırın... (Örn: Sözleşmede kararlaştırılan yıllık temerrüt faizi oranı %150 olarak belirlenmiştir.)"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{ marginTop: '16px', width: '100%', padding: '12px', backgroundColor: loading ? '#94a3b8' : '#2563eb', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: '6px', fontSize: '16px', cursor: loading ? 'not-allowed' : 'pointer', transition: 'background-color 0.2s' }}
          >
            {loading ? 'AI Agent Is Analyzing...' : 'Risk Analizini Başlat (Analyze Contract)'}
          </button>
        </div>
        
        <div style={{ flex: '1 1 500px', backgroundColor: '#fff', padding: '24px', borderRadius: '12px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', minHeight: '400px' }}>
          <h3 style={{ color: '#334155', marginTop: 0 }}>📋 Yapay Zeka Bulguları (AI Analysis Output)</h3>
          {result ? (
            <div>
              <div style={{ marginBottom: '20px' }}>
                <p style={{ fontWeight: 'bold', color: '#475569', marginBottom: '6px' }}>Mevzuat Referansı (RAG Context Matches):</p>
                <div style={{ backgroundColor: '#f8fafc', padding: '12px', borderRadius: '6px', border: '1px solid #e2e8f0', fontSize: '13px', color: '#334155' }}>
                  {result.references && result.references.length > 0 ? result.references[0] : "No direct regulatory matches found."}
                </div>
              </div>
              <div>
                <p style={{ fontWeight: 'bold', color: '#475569', marginBottom: '6px' }}>Risk Değerlendirme Raporu (Risk Report):</p>
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', color: '#0f172a', fontSize: '14px' }}>{result.report}</div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80%', color: '#94a3b8', textAlign: 'center', flexDirection: 'column' }}>
              <p style={{ fontSize: '48px', margin: '0 0 10px 0' }}>📄</p>
              <p>Sol tarafa hukuki bir metin yükleyip analizi başlattığınızda, LangGraph çoklu ajan denetim sonuçları burada canlı olarak listelenecektir.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;