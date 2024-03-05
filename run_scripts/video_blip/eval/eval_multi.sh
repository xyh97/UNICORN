for i in {11..0..-1}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --cfg-path lavis/projects/video_blip/eval/tg_activitynet_eval.yaml \
    --override-ckpt $i
done