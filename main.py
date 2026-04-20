import math
import re
import io
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from pypdf import PdfReader

app = FastAPI(title="Zenith Hybrid RAG Kernel v5.2", version="5.2.0")

class ZenithEngine:
    def __init__(self):
        self.store = []
        self.df = {}
        self.idf = {}
        self.total_chunks = 0
        self.total_terms = 0
        self.avg_length = 0.0
        self.stopwords = {'the','is','at','which','on','and','a','an','of','for','in','to',
                          'with','it','that','by','this','from','be','was','as','are','what',
                          'define','explain'}

    def clean(self, text: str) -> str:
        # Strip non-ASCII and normalize whitespace
        cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r'[^a-z0-9]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2 and t not in self.stopwords]
        # Basic stemming for plural 's'
        return [t[:-1] if t.endswith('s') and len(t) > 4 else t for t in tokens]

    def vectorize(self, tokens: List[str]) -> List[float]:
        # 256-bit orthogonal dense vector projection
        vec = [0.0] * 256
        for t in tokens:
            for i in range(256):
                h = 0
                for char in t:
                    h = ((h << 5) - h + ord(char)) & 0xFFFFFFFF
                vec[i] += (h % (i + 1))
        return vec

    def recalculate_idf(self):
        self.avg_length = self.total_terms / self.total_chunks if self.total_chunks > 0 else 0
        for t, freq in self.df.items():
            self.idf[t] = math.log(1 + (self.total_chunks - freq + 0.5) / (freq + 0.5))

    def add_node_to_store(self, text: str, tokens: List[str], meta: Dict[str, Any]):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            
        unique_tokens = set(tokens)
        for t in unique_tokens:
            self.df[t] = self.df.get(t, 0) + 1
            
        self.store.append({
            "text": text,
            "tokens": tokens,
            "tf": tf,
            "length": len(tokens),
            "vector": self.vectorize(tokens),
            "meta": meta
        })
        
        self.total_chunks += 1
        self.total_terms += len(tokens)

    def ingest_pdf_content(self, file_bytes: bytes, filename: str) -> int:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_processed = 0
        
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            text = self.clean(raw_text)
            chunks = re.split(r'[^.!?]+[.!?]+', text)
            chunks = [c for c in chunks if c.strip()]
            
            current_chunk = ""
            for j, chunk in enumerate(chunks):
                current_chunk += chunk + " "
                if (j + 1) % 3 == 0 or j == len(chunks) - 1:
                    clean_str = current_chunk.strip()
                    tokens = self.tokenize(clean_str)
                    
                    alpha_chars = len(re.sub(r'[^a-zA-Z]', '', clean_str))
                    total_chars = len(clean_str)
                    
                    if len(tokens) > 5 and total_chars > 0 and (alpha_chars / total_chars) > 0.4:
                        self.add_node_to_store(clean_str, tokens, {"source": filename, "page": i + 1, "type": "PDF"})
                    
                    current_chunk = ""
            pages_processed += 1
            
        self.recalculate_idf()
        return pages_processed

    def ingest_doi(self, doi: str) -> Dict[str, Any]:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract,authors,year"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("API request failed or DOI not found.")
            
        data = response.json()
        abstract_text = data.get("abstract") or "No abstract available."
        full_text = f"[TITLE: {data.get('title')}] [YEAR: {data.get('year')}] {abstract_text}"
        
        clean_text = self.clean(full_text)
        tokens = self.tokenize(clean_text)
        
        if tokens:
            self.add_node_to_store(clean_text, tokens, {"source": f"DOI: {doi}", "page": "API", "type": "SCHOLAR"})
            self.recalculate_idf()
            return data
        return {}

    def search(self, query: str) -> List[Dict[str, Any]]:
        q_tokens = self.tokenize(query)
        q_vec = self.vectorize(q_tokens)
        
        if not q_tokens:
            return []

        max_bm25 = 0.0
        results = []

        # Pass 1: Lexical & Semantic Scoring
        for chunk in self.store:
            # BM25 Lexical
            bm25 = 0.0
            for t in q_tokens:
                if t in chunk["tf"]:
                    num = chunk["tf"][t] * 2.5
                    den = chunk["tf"][t] + 1.5 * (0.25 + 0.75 * (chunk["length"] / self.avg_length))
                    bm25 += self.idf.get(t, 0.0) * (num / den)
            
            if bm25 > max_bm25:
                max_bm25 = bm25

            # Cosine Semantic
            dot = 0.0
            q_mag = 0.0
            n_mag = 0.0
            for i in range(256):
                dot += q_vec[i] * chunk["vector"][i]
                q_mag += q_vec[i] * q_vec[i]
                n_mag += chunk["vector"][i] * chunk["vector"][i]
                
            semantic = dot / (math.sqrt(q_mag) * math.sqrt(n_mag) + 1e-9)

            # Structural Definition Boost
            boost = 1.15 if re.search(r'(iff|defined as|definition|conflict happens|called|title:)', chunk["text"], re.IGNORECASE) else 1.0

            results.append({
                "text": chunk["text"],
                "meta": chunk["meta"],
                "bm25_raw": bm25,
                "semantic": semantic,
                "boost": boost
            })

        # Pass 2: Alpha-Beta Blending & True Sorting
        blended = []
        for r in results:
            norm_bm25 = (r["bm25_raw"] / max_bm25) if max_bm25 > 0 else 0.0
            # Strict Hallucination Lock: If Lexical is 0, score is 0
            raw_hybrid = ((norm_bm25 * 0.4) + (r["semantic"] * 0.6)) * r["boost"] if norm_bm25 > 0 else 0.0
            
            blended.append({
                "text": r["text"],
                "meta": r["meta"],
                "raw_score": raw_hybrid,
                "lexical_score": norm_bm25,
                "semantic_score": r["semantic"]
            })

        # Filter low signals and sort by raw mathematical resolution
        sorted_results = sorted([r for r in blended if r["raw_score"] > 0.05], key=lambda x: x["raw_score"], reverse=True)[:3]

        # Pass 3: Safe UI Representation Clamping
        final_output = []
        for r in sorted_results:
            r["score"] = min(0.999, r["raw_score"])
            final_output.append(r)
            
        return final_output

# Initialize Kernel
zenith = ZenithEngine()

# API Models
class DOIRequest(BaseModel):
    doi: str

class QueryRequest(BaseModel):
    query: str

# API Endpoints
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Must be a PDF file.")
    
    contents = await file.read()
    pages = zenith.ingest_pdf_content(contents, file.filename)
    
    return {
        "status": "success", 
        "pages_processed": pages, 
        "total_nodes_in_memory": zenith.total_chunks
    }

@app.post("/ingest/doi")
async def ingest_doi(req: DOIRequest):
    try:
        metadata = zenith.ingest_doi(req.doi)
        return {
            "status": "success", 
            "title": metadata.get("title"), 
            "total_nodes_in_memory": zenith.total_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search(req: QueryRequest):
    if zenith.total_chunks == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest documents first.")
        
    results = zenith.search(req.query)
    if not results:
        return {"status": "No match found", "results": []}
        
    return {"status": "success", "results": results}

@app.get("/status")
async def status():
    return {
        "engine": "Hybrid Lexical-Semantic",
        "nodes": zenith.total_chunks,
        "terms": zenith.total_terms
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
