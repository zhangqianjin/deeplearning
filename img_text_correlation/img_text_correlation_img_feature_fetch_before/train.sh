#!/bin/bash
train_DATA_DIR=data/ai_challenger_caption_train_20170902/
train_img_path=data/ai_challenger_caption_train_20170902/imgid2vec
#train_img_path=data/ai_challenger_caption_train_20170902/a
val_DATA_DIR=data/ai_challenger_caption_validation_20170910/
val_img_path=data/ai_challenger_caption_validation_20170910/imgid2vec
#val_img_path=data/ai_challenger_caption_validation_20170910/a
BERT_MODEL=/ssd1/zhangqianjin/pytorch_test/bert-base-chinese_file
OUTPUT_DIR=output
python3 train_net.py --train_data_dir ${train_DATA_DIR} \
                         --val_data_dir ${val_DATA_DIR} \
                         --train_img_path ${train_img_path} \
                         --val_img_path ${val_img_path} \
                         --bert_model ${BERT_MODEL} \
                         --output_dir ${OUTPUT_DIR} \
                         --train_batch_size 500 \
                         --max_seq_length 64 \
                         --do_train \
                         --do_eval \
                         --overwrite_output_dir    
