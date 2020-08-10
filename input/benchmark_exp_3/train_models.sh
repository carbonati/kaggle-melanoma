steps=20
keep_prob=1
batch_size=0
experiment_name="benchmark_exp_3"
sleep_sec=20
num_gpus=8

# export CUDA_VISIBLE_DEVICES=0,2,1,3
pkill -9 python

#python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_1.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
#sleep $sleep_sec
#pkill -9 python

#python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_2.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
#pkill -9 python
#sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_3.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_4.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_5.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_6.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_7.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
pkill -9 python
sleep $sleep_sec

python -u -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 9901 train.py --config_filepath input/$experiment_name/train_config_8.yaml -s $steps -p $keep_prob -e $experiment_name -b $batch_size --num_gpus $num_gpus  --fold_ids 1
sleep $sleep_sec
pkill -9 python

