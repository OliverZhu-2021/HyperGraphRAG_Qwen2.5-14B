import os
from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import bge_embedding_local

# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

rag = HyperGraphRAG(
    embedding_func=bge_embedding_local,
    chunk_token_size=512,
    chunk_overlap_token_size=32,
    node2vec_params={
        "dimensions": 1024,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    },
    working_dir=f"expr/example"
)

query_text = 'How strong is the evidence supporting a systolic BP target of 120-129 mmHg in elderly or frail patients, considering potential risks like orthostatic hypotension, the balance between cardiovascular benefits and adverse effects, and the feasibility of implementation in diverse healthcare settings?'

result = rag.query(query_text)
print(result)