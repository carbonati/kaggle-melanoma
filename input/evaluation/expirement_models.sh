keep_prob=1
batch_size=256
sleep_sec=10
num_gpus=1
exp_name="aug_exp_3"

for filename in "/root/workspace/kaggle-melanoma/models/$exp_name"/*; do
  python evaluate.py --config_filepath input/evaluation/evaluation_config_1.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus --model $filename
  pkill -9 python
  sleep $sleep_sec
done
#python evaluate.py --config_filepath input/evaluation/evaluation_config_1.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus  --fold_ids 0
#pkill -9 python
#sleep $sleep_sec
