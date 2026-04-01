from rag.retriever import MunicipalRetriever
import os
import sys
sys.path.append('.')

print("Testing RAG pipeline...")
ret = MunicipalRetriever()
if not os.path.exists("data/rag/faq_chunks.jsonl"):
    print("Generating FAQ data...")
    import rag.faq_data as faq_data
    faq_data.main()

ret.build_index("data/rag/faq_chunks.jsonl")
results = ret.retrieve("birth certificate", top_k=1)
print(results)
