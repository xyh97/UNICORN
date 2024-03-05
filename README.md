# LAVIS


## Installation

1. Creating conda environment

```bash
conda create -n unicorn python=3.10
conda activate unicorn
```
    
2. Build from source

```bash
cd UNICORN
pip install -r requirements.txt
pip install -e .
```

3. Unzip data & Train

download the folder ``/nfs/data/data/howto100m/annotations`` to your server in howto100m folder.

```bash
unzip activitynet.zip -d ./activitynet
unzip youcook2.zip -d ./youcook2
unzip qvhighlights.zip -d ./qvhighlights
unzip Charades_v1_480.zip -d ./charades

cd youcook2
mv video_1fps video_1fps_new
```


```
bash run_scripts/video_blip/train/pretrain_75frames_temp_aug.sh
```