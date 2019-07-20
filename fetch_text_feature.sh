#pip install pytorch-pretrained-bert
#下载bert-base-chinese

DATA_DIR=data/t
BERT_MODEL=../../bert-base-chinese_file
OUTPUT_DIR=output/test.tsv

python fetch_text_feature_pool_output.py --input_file ${DATA_DIR} \
                         --bert_model ${BERT_MODEL} \
                         --output_file ${OUTPUT_DIR} \
                         --batch_size 16 \
                         --max_seq_length 30
