# Session 19 — Script 04: Semantic Document Search with Sentence Transformers
# ===========================================================================
# This script upgrades the document query system from keyword matching (TF-IDF)
# to SEMANTIC understanding using pre-trained Sentence Transformer models.
#
# Real-World Scenario:
# Your company's employees are frustrated that the internal search engine only
# works when they use the EXACT words in the document. They want to type
# "automobile recall" and find documents about "car safety issues." TF-IDF
# cannot do this. Semantic Search can.
#
# This is the EMBEDDING step of a modern RAG (Retrieval-Augmented Generation) pipeline.
#
# Dependencies: pip install sentence-transformers scikit-learn numpy
#
# Note: The sentence-transformers library will download a pre-trained model
# (~90MB) on first run. Subsequent runs use the cached version.

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def create_knowledge_base():
    """
    An expanded knowledge base with documents that use DIFFERENT vocabulary
    to describe SIMILAR concepts — specifically designed to show where
    TF-IDF fails and semantic search succeeds.
    """
    documents = [
        {
            "id": "DOC_001",
            "title": "Vehicle Safety Recall Procedures",
            "content": (
                "When a manufacturing defect is discovered in automobiles, the company must "
                "initiate a voluntary recall. All affected vehicles must be inspected at authorized "
                "service centers within 60 days. Replacement parts are provided at no cost to the "
                "vehicle owner. The National Highway Traffic Safety Administration (NHTSA) must be "
                "notified within 5 business days of the defect confirmation. Failure to comply may "
                "result in penalties of up to $115 million per violation."
            ),
            "category": "Automotive Safety"
        },
        {
            "id": "DOC_002",
            "title": "Employee Mental Health and Wellbeing Program",
            "content": (
                "The company provides comprehensive psychological support services to all staff "
                "experiencing stress, anxiety, burnout, or emotional difficulties. Free confidential "
                "counseling sessions are available through our Employee Assistance Program (EAP). "
                "Each employee receives 12 therapy sessions per year at no cost. Managers are trained "
                "to recognize signs of workplace fatigue and are required to offer flexible arrangements. "
                "Mental health days do not count against sick leave allocations."
            ),
            "category": "Human Resources"
        },
        {
            "id": "DOC_003",
            "title": "Cloud Infrastructure Migration Playbook",
            "content": (
                "Transitioning on-premises workloads to cloud computing platforms requires careful "
                "planning. Phase 1 involves inventory assessment of all existing servers, databases, "
                "and networking configurations. Phase 2 implements a lift-and-shift migration for "
                "stateless applications. Phase 3 re-architects monolithic applications into "
                "microservices using containerized deployments on Kubernetes. All migrations must "
                "include a rollback plan and maintain 99.95% uptime SLA during the transition period."
            ),
            "category": "Information Technology"
        },
        {
            "id": "DOC_004",
            "title": "Sustainable Agriculture and Crop Rotation Guidelines",
            "content": (
                "Implementing crop rotation improves soil health and reduces pest dependency. "
                "The recommended 4-year rotation cycle is: Year 1 - Legumes (nitrogen fixation), "
                "Year 2 - Root vegetables (deep soil penetration), Year 3 - Brassicas (pest break), "
                "Year 4 - Cereals (biomass production). Cover crops should be planted during fallow "
                "periods to prevent erosion. Organic matter content should be maintained above 3% "
                "through composting and green manure applications."
            ),
            "category": "Agriculture"
        },
        {
            "id": "DOC_005",
            "title": "International Trade Compliance and Export Controls",
            "content": (
                "All cross-border shipments must comply with export control regulations including "
                "the Export Administration Regulations (EAR) and International Traffic in Arms "
                "Regulations (ITAR). Dual-use technology requires an export license from the Bureau "
                "of Industry and Security (BIS). Restricted party screening must be performed against "
                "the Consolidated Screening List (CSL) before any international transaction. Violations "
                "can result in criminal penalties of up to $1 million per infraction and 20 years imprisonment."
            ),
            "category": "Legal / Trade"
        },
        {
            "id": "DOC_006",
            "title": "Artificial Intelligence Model Governance Framework",
            "content": (
                "All machine learning models deployed in production must undergo a governance review "
                "before release. The review includes bias auditing using demographic parity and "
                "equalized odds metrics. Model cards documenting training data, performance metrics, "
                "and known limitations are mandatory. High-risk models (credit scoring, hiring, "
                "healthcare) require quarterly re-validation and drift monitoring. An AI ethics "
                "committee must approve any model that processes protected characteristics."
            ),
            "category": "AI Governance"
        },
        {
            "id": "DOC_007",
            "title": "Pharmaceutical Clinical Trial Protocol",
            "content": (
                "Phase III randomized controlled trials require a minimum of 3,000 participants "
                "across diverse demographic groups. The study uses double-blind methodology where "
                "neither the patient nor the administering physician knows whether the treatment is "
                "the experimental drug or placebo. Primary endpoints must be clinically significant "
                "with p-values below 0.05. Adverse event reporting to the FDA must occur within "
                "15 calendar days for serious events and 7 days for fatal or life-threatening events."
            ),
            "category": "Healthcare / Pharma"
        },
        {
            "id": "DOC_008",
            "title": "Renewable Energy Grid Integration Standards",
            "content": (
                "Solar and wind power installations connected to the national grid must comply with "
                "IEEE 1547 interconnection standards. Inverter-based resources must provide frequency "
                "response and voltage regulation capabilities. Battery energy storage systems (BESS) "
                "must maintain a minimum 4-hour discharge duration at rated capacity. Curtailment "
                "protocols activate when renewable generation exceeds 65% of total grid demand to "
                "maintain system stability. Power purchase agreements (PPAs) typically span 15-25 years."
            ),
            "category": "Energy"
        }
    ]
    return documents

def tfidf_search(documents, query, top_k=3):
    """Traditional TF-IDF keyword search for comparison."""
    texts = [doc['content'] for doc in documents]
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = similarities.argsort()[::-1][:top_k]
    
    return [(documents[i], similarities[i]) for i in ranked]

def semantic_search(documents, query, model, top_k=3):
    """Semantic search using sentence embeddings."""
    texts = [doc['content'] for doc in documents]
    
    # Encode all documents and query into dense embedding vectors
    doc_embeddings = model.encode(texts, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)
    
    # Cosine similarity in embedding space captures MEANING, not keywords
    similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
    ranked = similarities.argsort()[::-1][:top_k]
    
    return [(documents[i], similarities[i]) for i in ranked]

def run_comparison(documents, query, model, expected_doc_id, query_num):
    """Compare TF-IDF vs Semantic Search for a single query."""
    print(f"  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  Test {query_num}: {query[:56]:<56s}  ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Expected: {expected_doc_id}\n")
    
    # TF-IDF Results
    tfidf_results = tfidf_search(documents, query, top_k=3)
    tfidf_top = tfidf_results[0]
    tfidf_correct = tfidf_top[0]['id'] == expected_doc_id
    
    print(f"  📊 TF-IDF (Keyword Matching):")
    for doc, score in tfidf_results:
        marker = "→" if doc['id'] == expected_doc_id else " "
        print(f"    {marker} {doc['id']}: {doc['title'][:45]:<45s} [{score*100:.1f}%]")
    print(f"    Result: {'✅ CORRECT' if tfidf_correct else '❌ WRONG'}")
    
    print()
    
    # Semantic Search Results
    if model is not None:
        sem_results = semantic_search(documents, query, model, top_k=3)
        sem_top = sem_results[0]
        sem_correct = sem_top[0]['id'] == expected_doc_id
        
        print(f"  🧠 Semantic Search (Meaning-Based):")
        for doc, score in sem_results:
            marker = "→" if doc['id'] == expected_doc_id else " "
            print(f"    {marker} {doc['id']}: {doc['title'][:45]:<45s} [{score*100:.1f}%]")
        print(f"    Result: {'✅ CORRECT' if sem_correct else '❌ WRONG'}")
    else:
        sem_correct = None
    
    print()
    return tfidf_correct, sem_correct

def main():
    print_header("Semantic Search vs TF-IDF: Side-by-Side Comparison")
    print("This script demonstrates WHY semantic search (using neural embeddings)")
    print("is superior to keyword matching (TF-IDF) for document retrieval.\n")
    
    # ─── Stage 1: Load Knowledge Base ────────────────────────────────────
    documents = create_knowledge_base()
    print(f"📚 Knowledge Base: {len(documents)} documents loaded.\n")
    for doc in documents:
        print(f"  📄 {doc['id']}: [{doc['category']:>20s}] {doc['title']}")
    
    # ─── Stage 2: Load Semantic Model ────────────────────────────────────
    print_header("Stage 2: Loading Sentence Transformer Model")
    
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        
        print("  Loading 'all-MiniLM-L6-v2' model (384-dimensional embeddings)...")
        print("  (First run downloads ~90MB model. Cached for subsequent runs.)\n")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Demonstrate what an embedding looks like
        sample_embedding = model.encode(["Hello world"])
        print(f"  ✅ Model loaded successfully.")
        print(f"     Embedding dimensionality: {sample_embedding.shape[1]}")
        print(f"     Sample embedding (first 10 dims): {sample_embedding[0][:10].round(4)}")
        print(f"     Every text is now a point in {sample_embedding.shape[1]}-dimensional space.\n")
        
    except ImportError:
        print("  ⚠️  sentence-transformers is not installed.")
        print("     Install with: pip install sentence-transformers")
        print("     Running TF-IDF only mode for comparison.\n")
    
    # ─── Stage 3: The Challenge Queries ──────────────────────────────────
    # These queries deliberately use DIFFERENT vocabulary from the documents.
    # TF-IDF will struggle. Semantic search should handle them.
    
    print_header("Stage 3: Head-to-Head Comparison")
    print("  Each query uses SYNONYMS and PARAPHRASES — not the exact words")
    print("  in the document. This is how real users actually search.\n")
    
    challenge_queries = [
        # (query, expected_document_id)
        (
            "car safety issues and product defects", 
            "DOC_001",
        ),
        (
            "workers feeling overwhelmed and needing psychological help",
            "DOC_002",
        ),
        (
            "moving our servers to the internet and using containers",
            "DOC_003",
        ),
        (
            "farming techniques to improve dirt quality naturally",
            "DOC_004",
        ),
        (
            "sending technology products to foreign countries legally",
            "DOC_005",
        ),
        (
            "making sure AI algorithms are fair and unbiased before launch",
            "DOC_006",
        ),
        (
            "testing new medicines on human patients before government approval",
            "DOC_007",
        ),
        (
            "connecting windmills and solar panels to the power network",
            "DOC_008",
        ),
    ]
    
    tfidf_correct_count = 0
    semantic_correct_count = 0
    
    for i, (query, expected_id) in enumerate(challenge_queries, 1):
        tfidf_ok, semantic_ok = run_comparison(documents, query, model, expected_id, i)
        if tfidf_ok:
            tfidf_correct_count += 1
        if semantic_ok:
            semantic_correct_count += 1
    
    # ─── Stage 4: Scorecard ──────────────────────────────────────────────
    print_header("FINAL SCORECARD")
    
    total = len(challenge_queries)
    
    print(f"  {'Method':<35} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-' * 55}")
    print(f"  {'TF-IDF (Keyword Matching)':<35} {tfidf_correct_count:>5}/{total} {tfidf_correct_count/total*100:>9.1f}%")
    
    if model is not None:
        print(f"  {'Semantic Search (Embeddings)':<35} {semantic_correct_count:>5}/{total} {semantic_correct_count/total*100:>9.1f}%")
        print(f"  {'-' * 55}")
        improvement = semantic_correct_count - tfidf_correct_count
        print(f"\n  🧠 Semantic Search correctly answered {improvement} more queries than TF-IDF.")
    else:
        print(f"\n  ⚠️  Install sentence-transformers to compare semantic search performance.")
    
    print(f"\n  💡 KEY TAKEAWAY:")
    print(f"     TF-IDF matches WORDS. It fails when users use synonyms.")
    print(f"     Semantic Search matches MEANING. 'car' ≈ 'automobile' ≈ 'vehicle'.")
    print(f"     Modern RAG pipelines use semantic search as the retrieval backbone.\n")
    
    # ─── Stage 5: How Embeddings Work (Visual Explanation) ───────────────
    if model is not None:
        print_header("Stage 5: Understanding Embedding Similarity")
        
        pairs = [
            ("car", "automobile"),
            ("car", "banana"),
            ("doctor", "physician"),
            ("doctor", "basketball"),
            ("machine learning", "artificial intelligence"),
            ("machine learning", "cooking recipe"),
        ]
        
        print(f"  {'Word A':<25} {'Word B':<25} {'Similarity':>10}")
        print(f"  {'-' * 62}")
        
        for word_a, word_b in pairs:
            emb_a = model.encode([word_a])
            emb_b = model.encode([word_b])
            sim = cosine_similarity(emb_a, emb_b)[0][0]
            
            bar = "█" * int(sim * 30)
            print(f"  {word_a:<25} {word_b:<25} {sim:>8.3f}  {bar}")
        
        print(f"\n  ✅ Semantically similar words cluster together in embedding space.")
        print(f"     This is WHY semantic search outperforms keyword matching.\n")

if __name__ == "__main__":
    main()
