import os
import sys
sys.path.append('.')

print("Running PDF ingestion...")
os.system("python rag/pdf_ingest.py")

if not os.path.exists("data/rag/faq_chunks.jsonl"):
    print("Running FAQ data generation...")
    os.system("python rag/faq_data.py")

from rag.retriever import MunicipalRetriever
ret = MunicipalRetriever()
ret.build_index("data/rag")
os.makedirs("artifacts", exist_ok=True)
ret.save("artifacts/rag_index")
results = ret.retrieve("birth certificate", top_k=2)

with open("test_results.txt", "w", encoding="utf-8") as f:
    f.write("Test Retrieval Results:\n")
    for r in results:
         f.write(f"Score: {r['score']}\nQ: {r['question']}\nA: {r['answer']}\n---\n")

print("Test complete.")
