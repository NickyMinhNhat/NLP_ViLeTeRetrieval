import numpy as np
import json
import torch
from tqdm import tqdm
from rank_bm25 import *
import argparse
import os
import pickle
import glob
from utils import bm25_tokenizer
from datetime import datetime

from sentence_transformers import SentenceTransformer, util

def encode_legal_data(legal_dict_json, model):
    print("Start encoding legal data.")
    doc_data = json.load(open(legal_dict_json))
    emb2_list = []
    for k, doc in tqdm(doc_data.items()):
        emb2 = model.encode(doc_data[k]["title"] + " " + doc_data[k]["text"])
        emb2_list.append(emb2)
    emb2_arr = np.array(emb2_list)
    return emb2_arr

def encode_question(question_data, model):
    print("Start encoding questions.")
    emb_quest_dict = {}
    for _, item in tqdm(enumerate(question_data)):
        question_id = item["question_id"]
        question = item["question"]
        emb_quest_dict[question_id] = model.encode(question)
    return emb_quest_dict

def load_encoded_legal_corpus(legal_data_path):
    print("Start loading legal corpus.")
    basename = os.path.basename(legal_data_path)
    extension = basename.split(".")[-1]
    if extension in ["pickle", "pkl"]:
        with open(legal_data_path, "rb") as f1:
            emb_legal_data = pickle.load(f1)
    else: 
        with open(legal_data_path, "rb") as fnpy: 
            emb_legal_data = np.load(fnpy)
    if isinstance(emb_legal_data, list): 
        emb_legal_data = emb_legal_data[0]
    print("emb_legal_data: ", emb_legal_data.shape, type(emb_legal_data))
    return emb_legal_data

def load_bm25(bm25_path):
    with open(bm25_path, "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25

def load_models(model_path):
    model = SentenceTransformer(model_path)
    return model

def load_question_json(question_path):
    with open(question_path, encoding="utf-8") as f: 
        question_data = json.load(f)
    return question_data

def predict(args):
    model_name = os.path.basename(args.model_path)
    print("Start loading model.")
    model = load_models(args.model_path)

    # load question from json file
    question_items = load_question_json(args.data_path)["items"]
    
    print("Number of questions: ", len(question_items))
    
    # load bm25 model 
    bm25 = load_bm25(args.bm25_path)

    # load corpus to search
    print("Load legal data.")
    with open(args.legal_data, "rb") as doc_refer_file:
        doc_refers = pickle.load(doc_refer_file)

    # load pre encoded for legal corpus
    if args.encoded_legal_data_path:
        emb_legal_data = load_encoded_legal_corpus(args.encoded_legal_data_path)
    else:
        emb_legal_data = encode_legal_data(args.legal_dict_json, model)
        assert isinstance(emb_legal_data, np.ndarray), "encoded_legal_data need to be the type of np.ndarray"
        saved_encoded_legal_path = os.path.join(args.output_dir, f"encoded_legal_data_{model_name}.npy")
        with open(saved_encoded_legal_path, "wb") as npyfile: 
            np.save(npyfile, emb_legal_data)
        print("emb_legal_data is saved as", saved_encoded_legal_path)
    
    # create lookup table 
    with open(args.legal_dict_json, "r", encoding="utf-8") as file_legal: 
        lookup_table = json.load(file_legal)
        

    # encode question for query
    question_embs = encode_question(question_items, model)

    # define top n for compare and range of score
    range_score = args.range_score

    pred_list = []

    print("Start calculating results.")
    for idx, item in tqdm(enumerate(question_items)):
        question_id = item["question_id"]
        question = item["question"]
        labels = item["relevant_articles"] # List[Dict[str]]
        
        tokenized_query = bm25_tokenizer(question)
        doc_scores = bm25.get_scores(tokenized_query)

        
        emb1 = question_embs[question_id]
        emb2 = emb_legal_data
        scores = util.cos_sim(emb1, emb2)
        
        cos_sim = scores.squeeze(0).numpy()
        if args.sqrt_bm25: 
            new_scores = np.sqrt(doc_scores)* cos_sim
        else: 
            new_scores = doc_scores * cos_sim
        max_score = np.max(new_scores)

        new_scores = torch.tensor(new_scores)
        final_scores, indices = torch.topk(new_scores, args.topk)
        map_ids = [int(idx) for score, idx in zip(final_scores, indices) if float(score) >= max_score - range_score and float(score) <= max_score]
            
        pred_dict = {}
        pred_dict["question_id"] = question_id
        pred_dict["labels"] = labels
        pred_dict["relevant_articles"] = []
        
        # post processing character error
        dup_ans = []
        for idx, idx_pred in enumerate(map_ids):
            pred = doc_refers[idx_pred]
            law_id = pred[0]
            article_id = pred[1]
            
            if law_id.endswith("nd-cp"):
                law_id = law_id.replace("nd-cp", "nđ-cp")
            if law_id.endswith("nđ-"):
                law_id = law_id.replace("nđ-", "nđ-cp")
            if law_id.endswith("nð-cp"):
                law_id = law_id.replace("nð-cp", "nđ-cp")
            if law_id == "09/2014/ttlt-btp-tandtc-vksndtc":
                law_id = "09/2014/ttlt-btp-tandtc-vksndtc-btc"
            if law_id + "_" + article_id not in dup_ans:
                dup_ans.append(law_id + "_" + article_id)
                pred_dict["relevant_articles"].append(
                    {
                        "law_id": law_id, 
                        "article_id": article_id, 
                        "text": lookup_table[f"{law_id}_{article_id}"]
                    }
                )
        pred_list.append(pred_dict)
    
    return pred_list
def extract_pred_label(pred_list): 
    query_results = []
    query_ground_truth = []
    for prediction in pred_list: 
        pred = prediction["relevant_articles"]
        label = prediction["labels"] # List[dict]
        query_results.append([r["law_id"] +"_"+ r["article_id"] for r in pred])
        query_ground_truth.append([l["law_id"] + "_" + l["article_id"] for l in label])
    assert len(query_results) == len(query_ground_truth) == len(pred_list), \
        f"Length of query results: {len(query_results)}; length of ground truth {len(query_ground_truth)}" \
        f" length of total: {len(pred_list)}"
    
    return query_results, query_ground_truth
    
def calculate_recall_precision(predictions, references): 
    precisions = []
    recalls = []

    for i in range(len(predictions)): 
        predicted = predictions[i] 
        actual = references[i]

        true_positives = len(set(predicted).intersection(set(actual)))
        false_positives = len(set(predicted).difference(set(actual)))
        false_negatives = len(set(actual).difference(set(predicted)))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls 

def compute_f2score(pred_list): 
    predictions, references = extract_pred_label(pred_list=pred_list)
    precisions, recalls = calculate_recall_precision(predictions=predictions, references=references)
    f2_scores = [(5*(p+0.0001)*(r+0.0001))/(4*(p+0.0001)+(r+0.0001)) for p, r in zip(precisions, recalls)]
    f2_avg = sum(f2_scores) / len(f2_scores)
    
    return f2_scores, f2_avg



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="for loading model", required=True)
    parser.add_argument("--output_dir", default="./result", type=str, help="for saving results")
    parser.add_argument("--data_path", default="", type=str, help="for loading question")
    parser.add_argument("--legal_dict_json", default="generated_data/legal_dict.json", type=str)
    parser.add_argument("--bm25_path", default="saved_model/bm25_Plus_04_06_model_full_manual_stopword", type=str)
    parser.add_argument("--legal_data", default="saved_model/doc_refers_saved", type=str, help="path to legal corpus for reference")
    parser.add_argument("--range-score", default=2.6, type=float, help="range of cos sin score for multiple-answer")
    parser.add_argument("--encoded_legal_data_path", default="", type=str, help="for legal data encoding")
    parser.add_argument("--topk", default=5, type=int, help="Number of retrieved documents")
    parser.add_argument("--sqrt_bm25", action="store_true", help="Use sqrt(bm25_score)*cos_sim")

    args = parser.parse_args()
    model_path_bn = os.path.basename(args.model_path)
    predictions = predict(args)
    f2_scores, f2_avg = compute_f2score(predictions)
    print(f"{model_path_bn} - f2 score: ", f2_avg)
    result = {
        "f2_scores": f2_scores, 
        "f2_avg": f2_avg
    }
    # extract result
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dt = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    if not args.sqrt_bm25:
        saved_file_path = os.path.join(args.output_dir,f'result_{model_path_bn}_{dt}.json')
        saved_score_path = os.path.join(args.output_dir,f'score_{model_path_bn}_{dt}.json')
    else: 
        saved_file_path = os.path.join(args.output_dir,f'result_{model_path_bn}_sqrtBM25_{dt}.json')
        saved_score_path = os.path.join(args.output_dir,f'score_{model_path_bn}_sqrt_BM25_{dt}.json')
    with open(saved_file_path, 'w', encoding="utf-8") as outfile:
        json.dump(predictions, outfile, ensure_ascii=False)
    with open(saved_score_path, "w") as scorefile: 
        json.dump(result, scorefile)
    print("Inference result is saved at", saved_file_path)
    print("Score is saved at", saved_score_path)

    