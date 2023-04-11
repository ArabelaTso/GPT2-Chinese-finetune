python eval.py \
  --device 0,1 \
  --length 900 \
  --model_config model/final_model/config.json \
  --tokenizer_path cache/vocab.txt \
  --model_path model/final_model \
  --stride 4 \