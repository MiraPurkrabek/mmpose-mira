#!/usr/bin/env bash

# IN_DATA="../../data/NSFW/NudeNet_data/sexy/test/"
IN_VIDEO="../../data/U19_SKV_MIL_08_01_2022_1st_period_synced_1min/U19_SKV_MIL_08_01_2022_E_1st_period_synced_1min.mp4"

DET_MODEL="mask2former"
POSE_MODEL="swin"

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
elif [ "$DET_MODEL" == "hrnet" ]; then
    # SOTA det
    DET_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py"
    DET_PTH="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth"
elif [ "$DET_MODEL" == "mspn" ]; then
    # SOTA det
    DET_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py"
    DET_PTH="https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth"
else
    # Default pose
    POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
    POSE_PTH="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
fi

OUT_DATA="$IN_DATA/output/DET_${DET_MODEL}_POSE_${POSE_MODEL}"
OUT_DATA="output/"

python demo/body3d_two_stage_video_demo.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py \
    https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth \
    --video-path $IN_VIDEO \
    --out-video-root $OUT_DATA \
    --rebase-keypoint-height \
    # --use-multi-frames --online

    # $DET_CFG \
    # $DET_PTH \
    # $POSE_CFG \
    # $POSE_PTH \

    # configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py \
    # https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth \
    
    # configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py \
    # https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth  \