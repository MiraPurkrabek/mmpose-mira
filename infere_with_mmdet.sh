#!/usr/bin/env bash

# IN_DATA="../../data/NSFW/NudeNet_data/sexy/test/"
# IN_DATA="../../data/NSFW/web"
# IN_DATA="../../data/pose_experiments"
IN_DATA="../../data/NSFW/nezavadova/video_samples/"

DET_MODEL="mask2former"
POSE_MODEL="hrnet-body"

PARADIGM="top-down"
# PARADIGM="bottom-up"

KPT_THR=0.0001
RADIUS=6

if [ "$PARADIGM" == "top-down" ]; then

    #################
    # DETECTION
    #################

    if [ "$DET_MODEL" == "htc" ]; then
        # SOTA det
        DET_CFG="configs/mmdet/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py"
        DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth"
    elif [ "$DET_MODEL" == "mask2former" ]; then
        # SOTA det
        DET_CFG="configs/mmdet/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic.py"
        DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth"
    else
        # Default det
        DET_CFG="demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
        DET_PTH="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    fi

    #################
    # POSE
    #################

    if [ "$POSE_MODEL" == "hrnet" ]; then
        # SOTA pose
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth"
    elif [ "$POSE_MODEL" == "swin" ]; then
        # SOTA pose
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth"
    elif [ "$DET_MODEL" == "mspn" ]; then
        # SOTA det
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth"
    else
        # Default pose
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
    fi

    OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_${POSE_MODEL}_00001"

    python demo/top_down_img_demo_with_mmdet.py \
        $DET_CFG \
        $DET_PTH \
        $POSE_CFG \
        $POSE_PTH \
        --img-root $IN_DATA \
        --out-img-root $OUT_DATA \
        --output-json \
        --output-heatmap \
        --radius $RADIUS \
        --kpt-thr $KPT_THR \

else

    #################
    # POSE
    #################

    if [ "$POSE_MODEL" == "higher-hrnet-wholebody" ]; then
        # SOTA det
        POSE_CFG="configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/higherhrnet_w48_coco_wholebody_512x512.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_wholebody_512x512_plus-934f08aa_20210517.pth"
    elif [ "$POSE_MODEL" == "higher-hrnet-body" ]; then
        # SOTA det
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth"
    elif [ "$POSE_MODEL" == "hrnet-wholebody" ]; then
        # SOTA det
        POSE_CFG="configs/wholebody/2d_kpt_sview_rgb_img/associative_embedding/coco-wholebody/hrnet_w48_coco_wholebody_512x512.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_wholebody_512x512_plus-4de8a695_20210517.pth"
    elif [ "$POSE_MODEL" == "hrnet-body" ]; then
        # SOTA det
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth"
    else
        # Default pose
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"
    fi
    
    OUT_DATA="$IN_DATA/output/POSE_${POSE_MODEL}"

    python demo/bottom_up_img_demo.py \
        $POSE_CFG \
        $POSE_PTH \
        --img-path $IN_DATA \
        --out-img-root $OUT_DATA \
        --radius $RADIUS \
        # --kpt-thr $KPT_THR \

fi


    

