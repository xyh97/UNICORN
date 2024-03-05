CUDA_VISIBLE_DEVICES=0,7 python -m torch.distributed.run --nproc_per_node=2 --master_port=29552 evaluate.py --cfg-path lavis/projects/video_blip/eval/eval_aug.yaml
