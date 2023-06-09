# vi_legal_text_retrieval
# Solution:

## Pipeline

We use the pipepline in the picture below:
<p align="center">
    <img src="figs/Pipeline_trainning.png">
</p>
Our pipeline is combination of BM25 and Sentence Transfromer.
Let us describe our approach briefly:

- Step 1: We trained a BM25 model for searching similar pair. We used BM25 to create negative sentence pairs for training Sentence Transformer in Step 3.
- Step 1: We trained Masked Language Model using legal corpus from training data. Our masked languague models are
```
VinAI/PhoBert-Large
FPTAI/ViBert
```
- Step 3: Train Sentence Transformer + Contrative loss with 4 settings:
```
1. MLM PhoBert Large -> Sentence Transformer
2. MLM ViBert -> Sentence Transformer
3. MLM PhoBert Large -> Condenser -> Sentence Transformer
4. MLM PhoBert Large -> Co-Condenser -> Sentence Transformer
```
- Step 4: Using 4 models from step 3 to generate corresponding hard negative sentences for training round 2 in step 5.
- Step 5: Training 4 above models round 2.
- Step 5: Ensemble 4 models obtained from step 5.
## Environment 
- Because this repo is private, get your token to clone it, download guidance is below:
```bash
git clone https://<your_token>@github.com/khang-nguyen2907/vi_legal_text_retrieval.git
cd vi_legal_text_retrieval
conda create -n capstone python==3.8.10 -y
conda activate capstone
pip install -r requirements.txt
```

## Data
- Create folder `zac2021-ltr-data` to hold dataset.
- Make sure you are in root dir `vi_legal_text_retrieval`
```bash
mkdir zac2021-ltr-data
cd ./zac2021-ltr-data
gdown --id 1FJ4RM8JtN0B1ZwgKALG23jOzf2ZauRJy
unzip ZaloAI_2021.zip
cd ../
```

## Create Folder

Create a new folder for generated data for training ``mkdir generated_data``
```bash
mkdir generated_data
```

## Train BM 25
To train BM25: ``python bm25_train.py``
Use load_docs to save time for later run: ``python bm25_train.py --load_docs``

To evaluate: ``python bm25_create_pairs.py``
This step will also create top_k negative pairs from BM25. We choose top_k= 20, 50
Pairs will be saved to: pair_data/

These pairs will be used to train round 1 Sentence Transformer model

## Create corpus:
Run ``python create_corpus.py``
This step will create:
- corpus.txt  (for finetune language model)
- cocondenser_data.json (for finetune CoCondenser model)

## Finetune language model using Huggingface
Pretrained model:
- viBERT: FPTAI/vibert-base-cased
- vELECTRA: FPTAI/velectra-base-discriminator-cased
- phobert-base: vinai/phobert-base
- phobert-large: vinai/phobert-large

$MODEL_NAME= phobert-large
$DATA_FILE= corpus.txt
$SAVE_DIR= /path/to/your/save/directory

Run the following cmd to train Masked Language Model:
```
python run_mlm.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $DATA_FILE \
    --do_train \
    --do_eval \
    --output_dir $SAVE_DIR \
    --line_by_line \
    --overwrite_output_dir \
    --save_steps 2000 \
    --num_train_epochs 20 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32
```

##  Train condenser and cocondenser from language model checkpoint
Original source code here: https://github.com/luyug/Condenser (we modified several lines of code to make it compatible with current version of transformers)

### Create data for Condenser:

```bash
cd ./Condenser/
mkdir generated_data
python helper/create_train.py \
        --tokenizer_name vinai/phobert-base \
        --file ../generated_data/corpus.txt \
        --save_to ./generated_data/ \
        --max_len 256
```

$MODEL_NAME checkpoint from finetuned language model

```bash
python run_pre_training.py \
        --output_dir saved_model_condenser \
        --model_name_or_path ../saved_model/phobert-base/checkpoint-66000 \
        --do_train \
        --save_steps 2000 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --fp16 \
        --warmup_ratio 0.1 \
        --learning_rate 5e-5 \
        --num_train_epochs 8 \
        --overwrite_output_dir \
        --dataloader_num_workers 32 \
        --n_head_layers 2 \
        --skip_from 6 \
        --max_seq_length 256 \
        --train_dir ./generated_data/ \
        --weight_decay 0.01 \
        --late_mlm
```


## Train cocodenser:
First, we create data for cocodenser

```
python helper/create_train_co.py \
    --tokenizer vinai/phobert-large \
    --file ../generated_data/cocondenser/corpus.txt.json \
    --save_to data/large_co/corpus.txt.json \
```

Run the following cmd to train co-condenser model:
```
python  run_co_pre_training.py   \
    --output_dir saved_model/cocondenser/   \
    --model_name_or_path $CODENSER_CKPT   \
    --do_train   \
    --save_steps 2000   \
    --model_type bert   \
    --per_device_train_batch_size 32   \
    --gradient_accumulation_steps 1   \
    --fp16   \
    --warmup_ratio 0.1   \
    --learning_rate 5e-5   \
    --num_train_epochs 10   \
    --dataloader_drop_last   \
    --overwrite_output_dir   \
    --dataloader_num_workers 32   \
    --n_head_layers 2   \
    --skip_from 6   \
    --max_seq_length 256   \
    --train_dir ../generated_data/cocondenser/   \
    --weight_decay 0.01   \
    --late_mlm  \
    --cache_chunk_size 32 \
    --save_total_limit 1
```

## Train Sentence Transformer
### Round 1: using negative pairs of sentence generated from BM25
For each Masked Language Model, we trained a sentence transformer corresponding to it
Run the following command to train round 1 of sentence bert model

Note: Use cls_pooling for condenser and cocodenser
```
python train_sentence_bert.py
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 1 \
    --num_val $NUM_VAL\
    --epochs 10\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```

here we pick $NUM_VAL is 50 * 20 and 50 * 50 for top 20 and 50 pairs data respectively

### Round 2: using hard negative pairs create from Round 1 model
- Step 1: Run the following cmd to generate hard negative pairs from round 1 model:
```
python hard_negative_mining.py \
    --model_path /path/to/your/sentence/bert/model\
    --data_path /path/to/the/lagal/corpus/json\
    --save_path /path/to/directory/to/save/neg/pairs\
    --top_k top_k_negative_pair
```
Here we pick top k is 20 and 50.
- Use the data generated from step 1 to train round 2 of sentence bert model for each model from round 1:
To train round 2, please use the following command:
```
python train_sentence_bert.py
    --pretrained_model /path/to/your/pretrained/mlm/model\
    --max_seq_length 256 \
    --pair_data_path /path/to/your/negative/pairs/data\
    --round 2 \
    --num_val $NUM_VAL\
    --epochs 5\
    --saved_model /path/to/your/save/model/directory\
    --batch_size 32\
```
Tips: Use small learning rate for model convergence

## Prediction
### For reproducing result.
To get the prediction, we use 4 2-round trained models with mlm pretrained is Large PhoBert, PhoBert-Large-Condenser,
Pho-Bert-Large-CoCondenser and viBert-based. Final models and their corresponding weights are below:
- 1 x PhoBert-Large-Round2: 0.1
- 1 x Condenser-PhoBert-Large-round2: 0.3
- 1 x Co-Condenser-PhoBert-Large-round2: 0.4
- 1 x FPTAI/ViBert-base-round2: 0.2

``doc_refers_saved.pkl`` and ``legal_dict.json`` are generated in traning bm25 process and create corpus, respectively.
We also provide a file to re-generate it before inference.
```
python3 create_corpus.py --data zac2021-ltr-data --save_dir generated_data
python3 create_doc_refers.py --raw_data zac2021-ltr-data --save_path generated_data
```
We also provide embedding vectors which is pre-encoded by ensemble model in ``encoded_legal_data.pkl``.
If you want to verified and get the final submission, please run the following command:
```
python3 predict.py --data /path/to/test/json/data --legal_data generated_data/doc_refers_saved.pkl --precode
```

If you already have ``encoded_legal_data.pkl``, run the following command:
```
python3 predict.py --data /path/to/test/json/data --legal_data generated_data/doc_refers_saved.pkl
```

### Just for inference

Run the following command
```
chmod +x predict.sh
./predict.sh
```

## post-processing techniques:

- fix typo of nd-cp
- multiply cos-sim score with score from bm25, we pick score-range = [max-score - 2.6, max-score] and pick top 5 sentences for a question with multiple answers .


## Methods used but not work

- Training Round 3 for Sentence Transformer.
- Pseudo Label: Improve our single model performace but hurt ensembel preformance.

# Web interface 
- Make sure you get all required dependencies from `requirements.txt`
- Suppose that you are in the root directory: `vi_legal_text_retrieval`
- Note that if `gdown` does not download anything, please re-run `pip install -U gdown`
```bash
cd ./app
```
- Create `checkpoint` folder to hold checkpoint of bm25 model 
```bash
mkdir checkpoint/
#download pretrained language model 
gdown --id 1F_ZHEf4hiSWTfkW7BgLgLRtNGS1Px5iO
unzip -d checkpoint/ language_model_ckpt.zip
```
- Create `database` folder for hold legal corpus and its embedding
```bash
mkdir database
#download data
gdown --id 1--lTjev2lOA5Lk2sapt52I-7o0DycQve
unzip -d database data_language_model.zip
cd database
#download encoded legal data 
gdown --id 1WQwUHsmxBhKOMbgvkV8Gmcu7XfVzwKLl
cd ../
```
- Download pretraind sentence-bert for retrieval
```bash 
gdown --id 1jY7VVYDnnn0rPq04-X9KGX4WN8W0n3ht
unzip sentence-bert.zip
```
- After download required packages above, make sure your working tree like this: 
```
vi_legal_text_retrieval
│
└───app
│   │   app.py
│   │   ...
│   │
│   └───checkpoint
│   |   │   bm25_Plus_04_06_model_full_manual_stopword
│   |   │   ...
|   |
|   |___database
|   |   |   doc_refers_saved
|   |   |   documents_manual
|   |   |   encoded_legal_data.pkl
|   |   |   legal_dict.json
|   |
|   |___sentence-bert
|   |   |   ...
│   
|   README.md
|   ...
```
- Run the web 
```bash
python app.py
```
# Inference 
- Prepare database and checkpoint. Choose one of checkpoints [here](https://drive.google.com/drive/folders/1ruqGu3Lc43nMuasvBH-Gew6OxHttDjGP?fbclid=IwAR32_rKxfMwntLfOpKByPArj9p-6PawubYX3BZ-VFssQyW1VT2KBDAV-osU)
```bash
cd app/
# download database for retrieval 
gdown https://drive.google.com/drive/folders/1K_9PxD9vMZNpW3cjBaiV7oAEqeYWoRGP -O ./database --folder
# download checkpoint 
gdown <check_point_shared_drive> -0 ./sentence-bert-round2 --folder 
```
- Run chatbot UI, connect to `localhost:5000` or `http://127.0.0.1:5000` to open the app
```python 
python app.py
```
# Reference 
- CuongNN218: https://github.com/CuongNN218/zalo_ltr_2021

