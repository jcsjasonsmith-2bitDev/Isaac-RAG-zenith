# Isaac-RAG-zenith
Deterministic Alpha-Beta Fusion (0.4 \text{BM25} + 0.6 \text{Semantic}) to solve vector saturation and eliminate hallucinations in scientific data.


# ISAAC-497 Bounty Submission: Zenith Unified Pipeline v5.2

## Executive Summary
This submission fulfills the requirements for the ISAAC-497 Scientific Document Intelligence bounty. Zenith v5.2 is a zero-dependency, browser-native Hybrid Retrieval-Augmented Generation (RAG) kernel. It eschews heavy, hallucination-prone cloud LLM wrappers in favor of a deterministic, client-side Alpha-Beta Fusion architecture, guaranteeing absolute research integrity, privacy, and auditability.

## Core Architectural Innovations

### 1. True Mathematical Resolution (No-Clamp Sorting)
Traditional RAG implementations frequently suffer from "Vector Saturation," where multiple disparate chunks achieve >90% similarity, destroying ranking resolution. Zenith v5.2 decouples the mathematical sorting algorithm from UI clamping. It utilizes raw, floating-point evaluation to establish an absolute ranking order before normalizing for user presentation, eliminating three-way ties and semantic bloat.

### 2. Reciprocal Rank Fusion (Alpha-Beta Blending)
To eliminate the "Lexical Blindspot" (where algorithms fail to understand semantic intent) and the "Semantic Hallucination" (where models return conceptually similar but textually unverified data), Zenith implements a 40/60 Hybrid Fusion mechanism:
* **Lexical Anchor (40%):** Utilizes the Okapi BM25 probabilistic framework. This calculates strict Inverse Document Frequency (IDF) and applies Length Normalization (b=0.75, k1=1.5). If a queried term does not exist in the document, BM25 returns a strict `0.0`, triggering an absolute veto that prevents the system from hallucinating an answer.
* **Semantic Brain (60%):** Utilizes a custom 256-bit orthogonal dense vector projection. This evaluates the conceptual neighborhood of the query, allowing the engine to distinguish between introductory noise and formal definitions (bolstered by a 1.15x structural definition multiplier).

### 3. Unified Ingestion Engine
The pipeline treats local, highly structured PDFs and remote API metadata as equal citizens in the vector space:
* **Local Parsing:** Integrated `PDF.js` Web Worker implementation for asynchronous, non-blocking extraction of multi-column scientific layouts. It includes a dynamic alpha-ratio filter (0.4) to strip tabular noise and mathematical glyph corruption prior to vectorization.
* **External Bridge:** Direct integration with the Semantic Scholar Graph API, allowing dynamic resolution of Digital Object Identifiers (DOIs) to fetch and vectorize peer-reviewed abstracts and authorship metadata.

### 4. Audit-Ready Provenance
Scientific viability requires absolute transparency. Zenith features a discrete "Provenance Stream" that maps every retrieval back to its explicit origin (e.g., `[PDF: Pg 2]` or `[SCHOLAR: API]`). It exposes the underlying mathematics to the researcher, displaying the exact Lexical vs. Semantic ratios that drove the retrieval decision.

## Deployment Instructions
The entire application is contained within a single, portable HTML file with zero backend dependencies. 
1. Launch `index.html` in any modern WebGL-compliant browser.
2. Upload a standard multi-column scientific PDF (e.g., arXiv:2007.03575v1).
3. Query the interface using strict academic terminology. 
