keep_prob=1
batch_size=256
sleep_sec=10
num_gpus=8
exp_name="benchmark_exp_6"

for filename in "/workspace/models/$exp_name"/*; do
  python evaluate.py --config_filepath input/evaluation/evaluation_config_1.yaml -e $exp_name -p $keep_prob -b $batch_size --num_gpus $num_gpus --model $filename
  pkill -9 python
  sleep $sleep_sec
done
#python evaluate.py --config_filepath input/evaluation/evaluation_config_1.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus  --fold_ids 0
#pkill -9 python
#sleep $sleep_sec
