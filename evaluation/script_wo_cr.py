import json
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_source', default='medical')
args = parser.parse_args()
data_source = args.data_source

if __name__ == "__main__":
    # 打开 datasets/2WikiMultiHopQA/raw/qa_test.json
    with open(f"results/HyperGraphRAG/{data_source}/test_context.json", encoding='utf-8') as f:
        data = json.load(f)
    questions = [item['question'] for item in data]
    for d,q in tqdm(zip(data, questions)):
        d['context'] = d['context'].split("-----Sources-----")[0]
        
    save_dir = f"results/wo_CR/{data_source}/test_context.json"
    if os.path.exists(save_dir):
        os.remove(save_dir)
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
        
    with open(f"results/wo_CR/{data_source}/test_context.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to {save_dir}")