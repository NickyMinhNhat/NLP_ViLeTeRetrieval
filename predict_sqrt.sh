#!/bin/bash

MODEL_PATH01="/root/capstone/vi_legal_text_retrieval/New_train_val/3GB_FULL/3GB_FULL_R2/sbr2_3gb_full"
MODEL_PATH02="/root/capstone/vi_legal_text_retrieval/New_train_val/100MB_BASE/sbr2_100mb_base"
MODEL_PATH03="/root/capstone/vi_legal_text_retrieval/New_train_val/300MB_FULL/300MB_FULL_R2/sbr2_300mb_full"
MODEL_PATH04="/root/capstone/vi_legal_text_retrieval/New_train_val/300MB_LITE/300MB_LITE_R2/sbr2_300mb_lite"

ENCODED_LEGAL_DATA_PATH01="/root/capstone/vi_legal_text_retrieval/inference_metric/result/encoded_legal_data_sbr2_3gb_full.npy"
ENCODED_LEGAL_DATA_PATH02="/root/capstone/vi_legal_text_retrieval/inference_metric/result/encoded_legal_data_sbr2_100mb_base.npy"
ENCODED_LEGAL_DATA_PATH03="/root/capstone/vi_legal_text_retrieval/inference_metric/result/encoded_legal_data_sbr2_300mb_full.npy"
ENCODED_LEGAL_DATA_PATH04="/root/capstone/vi_legal_text_retrieval/inference_metric/result/encoded_legal_data_sbr2_300mb_lite.npy"

OUTPUT_DIR="/root/capstone/vi_legal_text_retrieval/inference_metric/result/"
DATA_PATH="/root/capstone/vi_legal_text_retrieval/inference_metric/data/val_answer_question.json"
LEGAL_DICT_JSON="/root/capstone/vi_legal_text_retrieval/generated_data/legal_dict.json"
BM25_PATH="/root/capstone/vi_legal_text_retrieval/saved_language_model/bm25_Plus_04_06_model_full_manual_stopword"
LEGAL_DATA="/root/capstone/vi_legal_text_retrieval/saved_language_model/doc_refers_saved"
ENCODED_LEGAL_DATA_PATH="/root/capstone/vi_legal_text_retrieval/generated_data/encoded_legal_data.pkl"
RANGE_SCORE=2.6
TOP_K=5


echo Running $MODEL_PATH01

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH01  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --topk $TOP_K \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH01 \
                            --sqrt_bm25



echo Running $MODEL_PATH02

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH02  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --topk $TOP_K \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH02 \
                            --sqrt_bm25



echo Running $MODEL_PATH03

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH03  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --topk $TOP_K \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH03 \
                            --sqrt_bm25



echo Running $MODEL_PATH04

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH04  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --topk $TOP_K \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH04 \
                            --sqrt_bm25