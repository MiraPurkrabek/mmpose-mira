# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np

import mmcv
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

keypoints_names = [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ]


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--out-txt-root',
        type=str,
        default='',
        help='Root of the output json file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    estimated_poses = pd.DataFrame()
    for i in mmcv.track_iter_progress(range(len(img_keys))):

        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if len(ann_ids) == 0:
            continue

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
        
        if args.out_txt_root != '':
            bboxes = np.zeros((1, 4))
            keypoints = np.zeros((1, 17, 3))
            for pose in pose_results:
                bboxes = np.concatenate([bboxes, pose['bbox'][None, :]])
                keypoints = np.concatenate([keypoints, pose['keypoints'][None, :, :]])
            
            bboxes = bboxes[1:, :]
            keypoints = keypoints[1:, :, :]

            ann_dict = {
                "id": ann_ids,
                "image_id": [image_id for _ in range(len(ann_ids))],
            }

            for bbox, letter in zip(bboxes.T, "ltwh"):
                ann_dict["bbox_{}".format(letter)] = bbox
            
            for kpt_i in range(keypoints.shape[1]):
                name = keypoints_names[kpt_i]
                ann_dict[name+"_x"] = keypoints[:, kpt_i, 0]
                ann_dict[name+"_y"] = keypoints[:, kpt_i, 1]

            ann_df = pd.DataFrame(ann_dict)
            estimated_poses = pd.concat([estimated_poses, ann_df], ignore_index=True)

        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=args.show,
            out_file=out_file)

    if args.out_txt_root != '':
        os.makedirs(args.out_txt_root, exist_ok=True)
        out_txt_path = osp.join(args.out_txt_root, "{}_poses.csv".format(
            osp.splitext(osp.basename(args.json_file))[0]
        ))
        estimated_poses.to_csv(out_txt_path, index=False, float_format="%.4f")


if __name__ == '__main__':
    main()
