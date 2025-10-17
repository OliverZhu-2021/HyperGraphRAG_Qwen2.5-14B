import os
import json
from hypergraphrag import HyperGraphRAG
from hypergraphrag.llm import bge_embedding_local

# os.environ["OPENAI_API_KEY"] = "PLACE YOUR OPENAI API KEY HERE"

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

with open(f"example_contexts.json", mode="r", encoding='utf-8') as f:
    unique_contexts = json.load(f)
    
rag.insert(unique_contexts)