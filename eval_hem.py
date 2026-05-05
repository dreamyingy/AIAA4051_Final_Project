from transformers import AutoConfig, AutoModelForSequenceClassification
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
parser.add_argument(
    "--hhem_model_path",
    default="models/hallucination_evaluation_model",
    type=str,
    help="Path to the local HHEM model directory.",
)
parser.add_argument(
    "--flan_model_path",
    default="models/flan-t5-base",
    type=str,
    help="Path to the local flan-t5-base directory required by HHEM.",
)
args = parser.parse_args()

dataset = args.dataset
device = args.device
load_path = f"results/{dataset}/prediction.pkl" 
save_path = f"results/{dataset}/correctness.json"

# Sentences we want sentence embeddings for
chunks = load_pickle_file(load_path)
results = [x for chunk in chunks for x in chunk]
config = AutoConfig.from_pretrained(args.hhem_model_path, trust_remote_code=True)
config.foundation = args.flan_model_path
model = AutoModelForSequenceClassification.from_pretrained(
    args.hhem_model_path,
    config=config,
    trust_remote_code=True,
)
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
