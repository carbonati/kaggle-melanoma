keep_prob=1
batch_size=0
sleep_sec=10
num_gpus=8

python evaluate.py --config_filepath input/evaluation/evaluation_config_1.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
pkill -9 python
sleep $sleep_sec

#python evaluate.py --config_filepath input/evaluation/evaluation_config_2.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_3.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_4.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_5.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_6.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_7.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
#
#python evaluate.py --config_filepath input/evaluation/evaluation_config_7.yaml -p $keep_prob -b $batch_size --num_gpus $num_gpus
#pkill -9 python
#sleep $sleep_sec
