import os
import sys

# Add current directory to path so it can find 'rag' folder
sys.path.append('.')

from rag.retriever import MunicipalRetriever

print("Loading RAG Index from data/rag ...")
ret = MunicipalRetriever()
# Loads both faq_chunks and pdf_chunks automatically
ret.build_index("data/rag")

print("\n" + "="*40)
print("=== Interactive RAG Tester ===")
print("Type 'quit' or 'exit' to stop.")
print("="*40 + "\n")

while True:
    try:
        query = input("Enter your query: ")
        if not query.strip():
            continue
        if query.lower() in ['quit', 'exit']:
            break
            
        print(f"\nSearching for: '{query}'...")
        results = ret.retrieve(query, top_k=3)
        
        if not results:
            print("No matches found.")
        else:
            for i, r in enumerate(results, 1):
                print(f"\n[Result {i}] | Score: {r['score']:.4f}")
                print(f"Source Context: {r['question']}")
                print(f"Content:\n{r['answer']}")
                print("-" * 50)
                
    except KeyboardInterrupt:
        break

print("\nExited.")
