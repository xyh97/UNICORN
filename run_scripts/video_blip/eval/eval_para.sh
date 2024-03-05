CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=29552 evaluate.py --cfg-path lavis/projects/video_blip/eval/eval_para.yaml
