# Session 19 — Script 03: Document Query System Using TF-IDF
# ===========================================================
# This script builds a complete Document Query System that lets users
# ask natural language questions and retrieves the most relevant document
# from a knowledge base using TF-IDF vectorization and cosine similarity.
#
# Real-World Scenario:
# You are building an internal search engine for a law firm. Lawyers need
# to type questions like "What are the penalties for contract breach?" and
# the system must return the most relevant legal document from the firm's
# knowledge base — ranked by mathematical relevance.
#
# This is the FOUNDATIONAL building block of Retrieval-Augmented Generation (RAG).
#
# Dependencies: pip install scikit-learn pandas numpy

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def print_header(title):
    print("\n" + "=" * 70)
    print(f"{title.center(70)}")
    print("=" * 70 + "\n")

def create_knowledge_base():
    """
    Simulate a corporate knowledge base with documents spanning
    multiple domains (HR, Legal, IT, Finance, Operations).
    In production, these would be loaded from a database, PDF parser,
    or document management system.
    """
    documents = {
        "HR_Employee_Leave_Policy": {
            "title": "Employee Leave Policy (HR-2024-001)",
            "content": (
                "All full-time employees are entitled to 21 days of paid annual leave per calendar year. "
                "Leave must be requested at least 14 days in advance through the HR portal. "
                "Unused leave may be carried over up to a maximum of 5 days into the next calendar year. "
                "Sick leave requires a medical certificate if the absence exceeds 2 consecutive days. "
                "Maternity leave is 16 weeks paid. Paternity leave is 4 weeks paid. "
                "Emergency leave of up to 3 days may be granted for bereavement or natural disasters."
            ),
            "department": "Human Resources"
        },
        "IT_Password_Security": {
            "title": "Password and Authentication Policy (IT-SEC-007)",
            "content": (
                "All employees must use passwords of at least 12 characters including uppercase, lowercase, "
                "numbers, and special characters. Passwords must be changed every 90 days and cannot repeat "
                "any of the last 10 passwords. Multi-factor authentication (MFA) is mandatory for all systems "
                "containing customer data or financial records. Shared accounts are strictly prohibited. "
                "VPN access requires both a certificate and a hardware security key (YubiKey). "
                "Report any suspected credential compromise to the Security Operations Center within 1 hour."
            ),
            "department": "Information Technology"
        },
        "Finance_Expense_Reimbursement": {
            "title": "Expense Reimbursement Guidelines (FIN-2024-003)",
            "content": (
                "Business expenses must be submitted within 30 days of the transaction date. "
                "All claims above $50 require an original receipt or digital scan. "
                "Travel expenses are reimbursed at economy class rates for flights under 6 hours. "
                "Business class is permitted for international flights exceeding 6 hours with director approval. "
                "Daily meal allowance is capped at $75 domestic and $100 international. "
                "Personal expenses, alcohol, and entertainment costs are not reimbursable. "
                "Mileage reimbursement for personal vehicle use is $0.67 per mile."
            ),
            "department": "Finance"
        },
        "Legal_Data_Protection": {
            "title": "Data Protection and Privacy Compliance (LEGAL-GDPR-001)",
            "content": (
                "The company processes personal data in compliance with GDPR, CCPA, and NDPA regulations. "
                "All customer data must be classified as either Public, Internal, Confidential, or Restricted. "
                "Restricted data includes financial records, health information, and biometric identifiers. "
                "Data retention periods: customer transaction records 7 years, employee records 3 years after "
                "termination, marketing consent records indefinitely. Data Subject Access Requests (DSARs) "
                "must be fulfilled within 30 days. Cross-border data transfers require Standard Contractual "
                "Clauses (SCCs) or adequacy decisions. Penalties for non-compliance can reach 4% of global revenue."
            ),
            "department": "Legal"
        },
        "Operations_Remote_Work": {
            "title": "Remote Work and Hybrid Policy (OPS-2024-012)",
            "content": (
                "Employees may work remotely up to 3 days per week with manager approval. "
                "Core collaboration hours are 10:00 AM to 2:00 PM in the employee's local timezone. "
                "A stable internet connection of at least 25 Mbps is required for remote work. "
                "The company provides a one-time $500 home office stipend for ergonomic equipment. "
                "All remote work must be conducted from the employee's registered home address or an "
                "approved co-working space. International remote work requires prior HR and Legal approval "
                "due to tax implications and employment law considerations."
            ),
            "department": "Operations"
        },
        "HR_Performance_Review": {
            "title": "Performance Review and Promotion Criteria (HR-PERF-002)",
            "content": (
                "Performance reviews are conducted bi-annually in June and December. "
                "Employees are rated on a 5-point scale across Technical Skills, Collaboration, "
                "Innovation, Leadership, and Delivery. A minimum average rating of 3.5 is required "
                "for promotion eligibility. Senior promotions (Director and above) require at least "
                "2 consecutive review cycles with ratings above 4.0 and a completed leadership development "
                "program. All promotion decisions must be documented with evidence-based justification "
                "and approved by the department VP and HR Business Partner."
            ),
            "department": "Human Resources"
        },
        "IT_Incident_Response": {
            "title": "Cybersecurity Incident Response Plan (IT-SEC-INCIDENT-003)",
            "content": (
                "A cybersecurity incident is any event that threatens the confidentiality, integrity, "
                "or availability of company information systems. Severity levels: P1 (Critical - active "
                "data breach), P2 (High - suspected intrusion), P3 (Medium - vulnerability detected), "
                "P4 (Low - policy violation). P1 incidents require immediate notification to the CISO, "
                "CEO, and Legal department within 15 minutes. Customer notification must occur within "
                "72 hours per GDPR requirements. All incidents must be logged in the SIEM system with "
                "root cause analysis completed within 5 business days."
            ),
            "department": "Information Technology"
        },
        "Finance_Budget_Approval": {
            "title": "Budget Approval and Procurement Process (FIN-PROC-005)",
            "content": (
                "All expenditures above $5,000 require a Purchase Order (PO) approved by the budget "
                "owner and Finance department. Expenditures above $25,000 require additional CFO approval. "
                "Vendor selection for contracts above $50,000 must follow a competitive bidding process "
                "with at least 3 qualified vendors. Sole-source justification requires VP and CFO sign-off. "
                "Software subscription renewals must be reviewed by IT Security 60 days before expiration. "
                "Capital expenditure requests above $100,000 require board approval at the quarterly meeting."
            ),
            "department": "Finance"
        }
    }
    return documents

def build_query_engine(documents):
    """
    Build a TF-IDF-based document retrieval engine.
    This converts all documents into mathematical vectors and creates
    an index for fast similarity search.
    """
    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id]['content'] for doc_id in doc_ids]
    doc_titles = [documents[doc_id]['title'] for doc_id in doc_ids]
    doc_departments = [documents[doc_id]['department'] for doc_id in doc_ids]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',      # Remove common words (the, is, at, etc.)
        max_features=5000,         # Vocabulary size
        ngram_range=(1, 2),        # Capture single words AND two-word phrases
        sublinear_tf=True          # Apply logarithmic TF scaling (dampens high-frequency terms)
    )
    
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    return {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'doc_ids': doc_ids,
        'doc_titles': doc_titles,
        'doc_departments': doc_departments,
        'doc_texts': doc_texts
    }

def query_documents(engine, query, top_k=3):
    """
    Query the document store with a natural language question.
    Returns the top-K most relevant documents ranked by cosine similarity.
    """
    # Convert the query into the same TF-IDF vector space
    query_vector = engine['vectorizer'].transform([query])
    
    # Calculate cosine similarity between query and all documents
    similarities = cosine_similarity(query_vector, engine['tfidf_matrix']).flatten()
    
    # Rank by similarity (descending)
    ranked_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(ranked_indices, 1):
        results.append({
            'rank': rank,
            'doc_id': engine['doc_ids'][idx],
            'title': engine['doc_titles'][idx],
            'department': engine['doc_departments'][idx],
            'similarity': similarities[idx],
            'content': engine['doc_texts'][idx]
        })
    
    return results

def explain_tfidf_match(engine, query, doc_idx):
    """
    Explain WHY a document matched by showing the top contributing TF-IDF terms.
    This gives transparency into the retrieval decision.
    """
    query_vector = engine['vectorizer'].transform([query]).toarray().flatten()
    doc_vector = engine['tfidf_matrix'][doc_idx].toarray().flatten()
    
    # Element-wise product reveals which terms contributed most to the similarity
    contributions = query_vector * doc_vector
    feature_names = engine['vectorizer'].get_feature_names_out()
    
    # Get top contributing terms
    top_indices = contributions.argsort()[::-1][:5]
    
    contributing_terms = []
    for idx in top_indices:
        if contributions[idx] > 0:
            contributing_terms.append({
                'term': feature_names[idx],
                'contribution': contributions[idx]
            })
    
    return contributing_terms

def main():
    print_header("Document Query System (TF-IDF + Cosine Similarity)")
    print("Scenario: An internal corporate knowledge base search engine.")
    print("Employees type questions in natural language and the system")
    print("retrieves the most relevant company policy document.\n")
    
    # ─── Stage 1: Build the Knowledge Base ───────────────────────────────
    print_header("Stage 1: Loading Corporate Knowledge Base")
    
    documents = create_knowledge_base()
    print(f"  📚 Loaded {len(documents)} documents:\n")
    
    for doc_id, doc in documents.items():
        word_count = len(doc['content'].split())
        print(f"    📄 [{doc['department']:>22s}] {doc['title']} ({word_count} words)")
    
    # ─── Stage 2: Build the TF-IDF Search Engine ────────────────────────
    print_header("Stage 2: Building TF-IDF Vector Index")
    
    engine = build_query_engine(documents)
    
    vocab_size = len(engine['vectorizer'].get_feature_names_out())
    print(f"  ✅ TF-IDF Index built successfully.")
    print(f"     Vocabulary size: {vocab_size} unique terms")
    print(f"     Document matrix shape: {engine['tfidf_matrix'].shape}")
    print(f"     Each document is now a {engine['tfidf_matrix'].shape[1]}-dimensional vector.\n")
    
    # ─── Stage 3: Run Queries ────────────────────────────────────────────
    print_header("Stage 3: Querying the Knowledge Base")
    
    test_queries = [
        "How many vacation days do I get per year?",
        "What are the rules for working from home?",
        "How do I change my password and set up two-factor authentication?",
        "What happens if there is a data breach?",
        "How do I get reimbursed for a business trip?",
        "What do I need for a promotion to director level?",
        "Can I buy software for my team? What approvals do I need?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"  ╔══════════════════════════════════════════════════════════════════╗")
        print(f"  ║  Query {i}: {query[:56]:<56s}  ║")
        print(f"  ╚══════════════════════════════════════════════════════════════════╝\n")
        
        results = query_documents(engine, query, top_k=3)
        
        for result in results:
            confidence = result['similarity'] * 100
            bar_length = int(confidence * 0.4)
            bar = "█" * bar_length
            
            marker = "🥇" if result['rank'] == 1 else ("🥈" if result['rank'] == 2 else "🥉")
            
            print(f"    {marker} Rank #{result['rank']}: {result['title']}")
            print(f"       Department: {result['department']}")
            print(f"       Confidence: {confidence:.1f}% {bar}")
            
            # Show WHY this document matched (XAI for search!)
            if result['rank'] == 1:
                top_result_idx = engine['doc_ids'].index(result['doc_id'])
                contributing = explain_tfidf_match(engine, query, top_result_idx)
                if contributing:
                    terms_str = ", ".join([f"'{t['term']}'" for t in contributing])
                    print(f"       Key matching terms: {terms_str}")
                
                # Show a snippet of the matched content
                snippet = result['content'][:150] + "..."
                print(f"       Preview: \"{snippet}\"")
            
            print()
        
        print()
    
    # ─── Stage 4: Demonstrating Limitations ──────────────────────────────
    print_header("Stage 4: TF-IDF Limitations (Why We Need Semantic Search)")
    
    tricky_queries = [
        ("Where can I find rules about taking time off?", "HR_Employee_Leave_Policy"),
        ("How do I protect customer information?", "Legal_Data_Protection"),
    ]
    
    print("  TF-IDF matches KEYWORDS, not MEANING. Watch what happens:\n")
    
    for query, expected_doc in tricky_queries:
        results = query_documents(engine, query, top_k=1)
        top_result = results[0]
        matched = "✅" if top_result['doc_id'] == expected_doc else "⚠️"
        
        print(f"    Query:    \"{query}\"")
        print(f"    Expected: {expected_doc}")
        print(f"    Got:      {top_result['doc_id']} ({top_result['similarity']*100:.1f}%)")
        print(f"    Status:   {matched}")
        print()
    
    print("  💡 Notice: TF-IDF struggles when the query uses SYNONYMS")
    print("     (e.g., 'time off' vs 'leave', 'protect' vs 'compliance').")
    print("     Script 04 solves this with Semantic Search using embeddings.\n")

if __name__ == "__main__":
    main()
