python -m torch.distributed.run --nproc_per_node=8 --master_port=29550 train.py --cfg-path lavis/projects/video_blip/train/pretrain_75frames_temp_aug.yaml