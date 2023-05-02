MODEL_PATH01=""
MODEL_PATH02=""
MODEL_PATH03=""
MODEL_PATH04=""
OUTPUT_DIR=""
DATA_PATH=""
LEGAL_DICT_JSON=""
BM25_PATH=""
LEGAL_DATA=""
ENCODED_LEGAL_DATA_PATH=""
RANGE_SCORE=2.6
TOP_K=5

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH01  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH01  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K \
                            --sqrt_bm25

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH02  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH02  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K \
                            --sqrt_bm25

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH03  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH03  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K \
                            --sqrt_bm25

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH04  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K

CUDA_VISIBLE_DEVICES=0 python3 predict.py \
                            --model_path $MODEL_PATH04  \
                            --output_dir $OUTPUT_DIR \
                            --data_path $DATA_PATH \
                            --legal_dict_json $LEGAL_DICT_JSON \
                            --bm25_path $BM25_PATH \
                            --legal_data $LEGAL_DATA \
                            --range-score $RANGE_SCORE \
                            --encoded_legal_data_path $ENCODED_LEGAL_DATA_PATH \
                            --topk $TOP_K \
                            --sqrt_bm25