from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import argparse
import torch
import json
from tqdm import tqdm
import pickle

def load_pickle_file(path): 
    objs = [] 
    with open(path, "rb") as f: 
        while True: 
            try: 
                objs.append(pickle.load(f)) 
            except EOFError: 
                break
    return objs

def append_pickle(obj, path):
    with open(path, "ab") as f:
        pickle.dump(obj, f)

parser = argparse.ArgumentParser(description="Evaluate prediction consistency with HHEM.")
parser.add_argument(
    "--dataset",
    type=str,
    choices=["sciq", "simple_questions_wiki", "nq", "truthfulQA"],
    required=True,
    help="Dataset name under results/.",
)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

dataset = args.dataset
device = args.device
load_path = f"results/{dataset}/prediction.pkl" 
save_path = f"results/{dataset}/correctness.json"

# Sentences we want sentence embeddings for
chunks = load_pickle_file(load_path)
results = [x for chunk in chunks for x in chunk]
tokenizer=AutoTokenizer.from_pretrained('models/flan-t5-base') # path to flan-t5-base
model = AutoModelForSequenceClassification.from_pretrained( 
    'models/hallucination_evaluation_model', trust_remote_code=True) # path to hallucination_evaluation_model
model.eval()
model.to(device)

scores = []
for item in tqdm(results):
    merged_prediction = item[0]['merged_prediction']
    merged_true_answer = item[0]['merged_true_answer']
    score = model.predict([(merged_true_answer, merged_prediction)])[0]
    scores.append(score.item())


with open(save_path, "w") as f:
    json.dump(scores, f) 
