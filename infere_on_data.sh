#!/usr/bin/env bash

CAMERAS=("E" "N" "S" "W")

MATCH="U19_SKV_MIL_08_01_2022_1st_period_synced_1min"
STAGE="U19_SKV_MIL_08_01_2022_1st_period_synced_1min_custom_GEOM"

for CAMERA in ${CAMERAS[*]}; do
    python3 demo/top_down_img_demo.py \
        configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py \
        https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth \
        --img-root data/$MATCH/camera_$CAMERA/ \
        --json-file data/$MATCH/${MATCH}_camera_${CAMERA}_COCO_anns.json \
        --out-img-root vis_results/$STAGE/top_down/camera_$CAMERA \
        --out-txt-root txt_results/$STAGE/top_down/camera_$CAMERA \
    
    # python demo/bottom_up_img_demo.py \
    #     configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
    #     https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
    #     --img-path data/$MATCH/$STAGE/ \
    #     --out-img-root vis_results/$STAGE/bottom_up/
done

