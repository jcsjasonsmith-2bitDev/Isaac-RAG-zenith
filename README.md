# Zenith Scientific Kernel v5.2
### Deterministic Hybrid RAG for High-Integrity Research

Zenith v5.2 is a specialized Document Intelligence engine built for scientific and research workflows. It addresses the hallucination gap in standard RAG pipelines by utilizing a deterministic Alpha-Beta Fusion architecture, ensuring that every AI claim is grounded in lexical reality before semantic inference is applied.

---

## Live Demo
The browser-native implementation is live and can be tested immediately:
**[Launch Zenith Scientific Kernel (Replit)](https://asset-manager--jcsjasonsmith.replit.app/)**

---

## Core Innovations

### 1. Strict Hallucination Lock (Lexical Veto)
Unlike standard vector-only searches that always return the least-wrong answer, Zenith utilizes a hard-veto mechanism. If the queried terms do not exist in the document (BM25 score = 0), the system suppresses semantic hallucination, returning a "No match found" status instead of a fabricated response.

### 2. Alpha-Beta Hybrid Fusion
The kernel employs a dual-stream ranking algorithm to solve Vector Saturation:
* **Lexical Anchor (40%):** Uses the Okapi BM25 framework for strict keyword relevance and term frequency.
* **Semantic Brain (60%):** Utilizes 256-bit orthogonal dense vector projections to understand conceptual context.
* **Mathematical Resolution:** Results are sorted using raw floating-point values to eliminate ranking ties and semantic bloat.

### 3. Audit-Ready Provenance
Scientific viability requires absolute transparency. Zenith features a discrete Provenance Stream that maps every retrieval back to its explicit origin (e.g., [PDF: Pg 2] or [SCHOLAR: API]). It exposes the underlying mathematics to the researcher, displaying the exact Lexical vs. Semantic ratios that drove the retrieval decision.

---

## Technical Comparison

| Feature | Standard LlamaIndex/RAG | Zenith v5.2 |
| :--- | :--- | :--- |
| **Integrity Control** | Probabilistic (Guessing) | **Deterministic (Lexical Veto)** |
| **Ranking Resolution** | Clamped/Vector Only | **Raw Floating-Point Hybrid** |
| **Data Privacy** | Cloud-Dependent | **Browser-Native / Local-First** |
| **Provenance** | Opaque Citation | **Deep Metadata Mapping** |
| **Layout Extraction** | Generic Text Scrapers | **Multi-Column Scientific Parser** |

---

## Architecture

The system is designed for maximum flexibility, offering both a zero-dependency browser implementation and a high-performance Python backend.

### Hybrid Retrieval Formula
The final ranking score $S$ for a document chunk is calculated as:

$$S = (0.4 \times \text{Norm}(\text{BM25}) + 0.6 \times \text{Semantic}) \times \text{Boost}_{\text{structural}}$$

Where $\text{Boost}_{\text{structural}}$ is a 1.15x multiplier applied to formal definitions and technical identifiers.

---

## Getting Started

### Option A: Browser-Native (Zero Setup)
1. Open index.html in any modern WebGL-compliant browser.
2. Ingest a scientific PDF via the Local Pipeline or fetch peer-reviewed metadata via the Semantic Scholar Bridge.
3. Query the index directly. All processing stays on the client side, ensuring data privacy.

### Option B: Python Backend (FastAPI)
1. Install dependencies:
   `pip install fastapi uvicorn pypdf requests`
2. Run the server:
   `python main.py`
3. Access the API:
   Navigate to http://localhost:8000/docs to view the interactive documentation for PDF ingestion and Hybrid Search endpoints.

---

## Features
* **Advanced PDF Parsing:** Specialized logic to handle multi-column layouts and strip mathematical glyph noise (Dynamic Alpha-Ratio Filter > 0.4).
* **Semantic Scholar Integration:** Resolve DOIs into vectorized abstracts and metadata dynamically via the Graph API.
* **High-Speed Indexing:** Optimized for rapid ingestion without compromising retrieval accuracy.
* **Privacy-First:** Designed for air-gapped environments or high-compliance research workflows.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Developed for ISAAC-497 | Scientific Document Intelligence**
