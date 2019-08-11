#head -n 5 t
#Our friends won't buy this analysis, let alone the next one we propose.
#One more pseudo generalization and I'm giving up.
#One more pseudo generalization or I'm giving up.
#The more we study verbs, the crazier they get.
#Day by day the facts are getting murkier.

python3 BertForMaskedLM_best_train.py  --train_corpus 't' --bert_model 'bert-base-uncased' --output_di
r 'output' --do_train  --train_batch_size 20  --max_seq_length 30
