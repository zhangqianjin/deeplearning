python3 -u train/train.py  --data_path data/${datetime}  --model_path  model/${datetime}

python3 -u predict/predict.py --gpu_num $value  --data_path data/${datetime} --model_path ../model/best_model/best_model --topN 500
