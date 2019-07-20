#pip install pytorch-pretrained-bert
#下载bert-base-chinese
#t 马晓旭意外受伤让国奥警惕 无奈大雨格外青睐殷家军记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨


DATA_DIR=data/t
BERT_MODEL=../../bert-base-chinese_file
OUTPUT_DIR=output/test.tsv

python fetch_text_feature_pool_output.py --input_file ${DATA_DIR} \
                         --bert_model ${BERT_MODEL} \
                         --output_file ${OUTPUT_DIR} \
                         --batch_size 16 \
                         --max_seq_length 30
