#!/usr/bin/env bash

CAMERA="N"
MATCH="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
STAGE="U19_SKV_MIL_08_01_2022_1st_period_synced_1min_custom_GEOM"

python3 demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root data/$MATCH/camera_$CAMERA/ \
    --json-file data/$MATCH/camera_${CAMERA}_bbs.json \
    --out-img-root vis_results/$STAGE/top_down/camera_$CAMERA \
    --out-txt-root txt_results/$STAGE/top_down/camera_$CAMERA \

# python demo/bottom_up_img_demo.py \
#     configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
#     https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
#     --img-path data/$MATCH/$STAGE/ \
#     --out-img-root vis_results/$STAGE/bottom_up/