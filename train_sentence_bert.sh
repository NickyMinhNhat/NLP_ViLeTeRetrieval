python train_sentence_bert1.py --pretrained_model "/root/capstone/vi_legal_text_retrieval/New_train_val/3GB_FULL/3GB_FULL_R1/sentence-bert"  --max_seq_length 256 --train_pair_data_path "/root/capstone/vi_legal_text_retrieval/New_train_val/New_Data_Pair/New_Data_Pair/neg_pair/save_pairs_vibert_top20_train.pkl" --val_pair_data_path "/root/capstone/vi_legal_text_retrieval/New_train_val/New_Data_Pair/New_Data_Pair/neg_pair/save_pairs_vibert_top20_val.pkl" --rounds 2 --num_val 1000 --epochs 5 --saved_model "/root/capstone/vi_legal_text_retrieval/New_train_val/3GB_FULL/3GB_FULL_R2/sentence-bert-2" --batch_size 32