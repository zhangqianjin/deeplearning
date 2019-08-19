train_DATA_DIR=train_sample
train_img_path=${train_DATA_DIR}/img
train_img_file=${train_DATA_DIR}/img2id
val_DATA_DIR=val_sample
val_img_path=${val_DATA_DIR}/img
val_img_file=${val_DATA_DIR}/img2id
BERT_MODEL=/pytorch_test/bert-base-chinese_file
OUTPUT_DIR=output
python3 train_net.py --train_data_dir ${train_DATA_DIR} \
                         --val_data_dir ${val_DATA_DIR} \
                         --train_img_path ${train_img_path} \
                         --train_img_file ${train_img_file} \
                         --val_img_path ${val_img_path} \
                         --val_img_file ${val_img_file} \
                         --bert_model ${BERT_MODEL} \
                         --output_dir ${OUTPUT_DIR} \
                         --train_batch_size 16 \
                         --max_seq_length 64 \
                         --do_train \
                         --do_eval \
                         --overwrite_output_dir    
