import os
import json
import time
from hypergraphrag import HyperGraphRAG, bge_embedding_local
import argparse

os.environ["OPENAI_API_KEY"] = open("openai_api_key.txt").read().strip()
os.environ["HUGGINGFACE_API_KEY"] = open("huggingface_api_key.txt").read().strip()

def insert_text(rag, file_path):
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract just the context strings from the new structure
    unique_contexts = [item["context"] for item in data]

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")

parser = argparse.ArgumentParser()
parser.add_argument("--cls", type=str, default="medical")
args = parser.parse_args()
cls = args.cls
WORKING_DIR = f"expr/{cls}"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

rag = HyperGraphRAG(
    working_dir=WORKING_DIR,
    embedding_func_max_async=32,
    llm_model_max_async=32,
    embedding_func=bge_embedding_local,
    chunk_token_size=512,
    chunk_overlap_token_size=64,
    node2vec_params={
        "dimensions": 1024,
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }
)

insert_text(rag, f"contexts/{cls}_contexts.json")