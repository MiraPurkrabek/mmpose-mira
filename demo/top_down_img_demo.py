# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
import json
import pandas as pd
import numpy as np
import cv2

import mmcv
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)

import torchvision.transforms as T
from PIL import Image


def _box2cs(box, image_size):
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    scale = scale * 1.25
    return center, scale

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
keypoints_names_short = [
                "nose",
                "l_eye",
                "r_eye",
                "l_ear",
                "r_ear",
                "l_shldr",
                "r_shldr",
                "l_elbw",
                "r_elbw",
                "l_wrst",
                "r_wrst",
                "l_hip",
                "r_hip",
                "l_knee",
                "r_knee",
                "l_ankle",
                "r_ankle"
            ]


def parse_args():
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
    return args


def main(args):
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    
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
    return_heatmap = hasattr(args, "output_heatmap") and args.output_heatmap

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
            out_file = os.path.join(args.out_img_root, "keypoints", f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
            os.makedirs(
                os.path.join(args.out_img_root,"keypoints"),
                exist_ok=True
            )
            if return_heatmap:
                out_general_heatmap_file = os.path.join(
                    args.out_img_root,
                    "heatmaps",
                    f'vis_{osp.splitext(osp.basename(image_name))[0]}_heatmap.jpg'
                )

                os.makedirs(
                    os.path.join(args.out_img_root,"heatmaps"),
                    exist_ok=True
                )

                out_general_heatmap_orig = os.path.join(
                    args.out_img_root,
                    f'vis_{osp.splitext(osp.basename(image_name))[0]}_heatmap_orig.jpg'
                )
                # out_heat_folder = os.path.join(
                #     args.out_img_root,
                #     "heatmaps",
                #     image['file_name'].split(".")[0]
                # )
                # os.makedirs(out_heat_folder, exist_ok=True)
        
        if args.out_txt_root != '':

            keypoints = np.zeros((1, 17, 3))
            for pose in pose_results:
                keypoints = np.concatenate([keypoints, pose['keypoints'][None, :, :]])
            
            keypoints = keypoints[1:, :, :]

            ann_dict = {
                "id": ann_ids,
            }

            for kpt_i in range(keypoints.shape[1]):
                name = keypoints_names[kpt_i]
                ann_dict[name+"_x"] = keypoints[:, kpt_i, 0]
                ann_dict[name+"_y"] = keypoints[:, kpt_i, 1]

            ann_df = pd.DataFrame(ann_dict)
            estimated_poses = pd.concat([estimated_poses, ann_df], ignore_index=True)

        # if not return_heatmap:
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

        if return_heatmap:
            # print()

            heatmap = np.mean(returned_outputs[0]['heatmap'], axis=0)
            general_heatmap = np.max(heatmap, axis=0)
            general_heatmap = ((general_heatmap - np.min(general_heatmap[:])) * 255)
            img = mmcv.image.imread(image_name)

            orig_h, orig_w, _ = img.shape

            required_h = 384
            required_w = 288

            center, scale = _box2cs([0, 0, orig_w, orig_h], (required_w, required_h))
            trans = get_affine_transform(center, scale, 0, (required_w, required_h))
            inv_trans = get_affine_transform(center, scale, 0, (required_w, required_h), inv=True)
            input_img = cv2.warpAffine(
                img,
                trans, (int(required_w), int(required_h)),
                flags=cv2.INTER_LINEAR
            )

            # scaled_height = required_h
            # scaled_width = int(orig_w * scaled_height / orig_h)
            # if scaled_width > required_w:
            #     scaled_width = required_w
            #     scaled_height = int(orig_h * scaled_width / orig_w)
            # lower_size = min(scaled_width, scaled_height)

            # img = T.Resize(size=int(lower_size))(img)
            
            # prepad_w, prepad_h = img.size

            # # print("Scaled:", scaled_width, scaled_height)
            # # print("Prepad:", prepad_w, prepad_h)
            
            # img = T.Pad((int((required_w-img.size[0])/2), int((required_h-img.size[1])/2)))(img)
            # # print("After pad:", img.size)
            # img = T.Resize(size=(required_h, required_w))(img)
            
            # img = np.array(img)

            general_heatmap_3c = np.zeros((orig_h, orig_w, 3))
            
            # general_heatmap_resize = cv2.resize(
            #     general_heatmap,
            #     (required_w, required_h),
            #     interpolation=cv2.INTER_CUBIC
            # )
            # general_heatmap_3c[:, :, 2] = general_heatmap_resize
            
            for kpt_channel in range(heatmap.shape[0]):
                htm = heatmap[kpt_channel, :, :].squeeze()
                htm = ((htm - np.min(htm[:])) * 255)
                intermediate_htm = cv2.resize(
                    htm,
                    (required_w, required_h),
                    interpolation=cv2.INTER_CUBIC,
                )
                big_htm = cv2.warpAffine(
                    intermediate_htm,
                    inv_trans, (int(orig_w), int(orig_h)),
                    flags=cv2.INTER_CUBIC
                )

                # general_heatmap_3c[:, :, int(kpt_channel%3)] += cv2.resize(
                #     htm,
                #     (required_w, required_h),
                #     interpolation=cv2.INTER_CUBIC,
                # )
                general_heatmap_3c[:, :, int(kpt_channel%3)] += big_htm

                color = [100, 100, 100]
                color[int(kpt_channel%3)] = 255
                # coors = (
                #     np.argmax(big_htm, axis=1).astype(int)[0],
                #     np.argmax(big_htm, axis=0).astype(int)[0]
                # )

                coors = np.unravel_index(big_htm.argmax(), big_htm.shape)

                peak = big_htm[coors]

                # Only show those keypoints with ppt higher than 10%
                if peak/255 > 0.1:

                    # print()
                    # print(image_name)
                    # print("\t{:s}: \t\t {:.4f}".format(keypoints_names[kpt_channel], peak))

                    general_heatmap_3c = cv2.putText(
                        general_heatmap_3c,
                        "{:s} ({:.1f})".format(keypoints_names_short[kpt_channel], peak/255),
                        coors[::-1],
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = color,
                        thickness = 2,
                    )

            
            general_heatmap_3c = general_heatmap_3c / np.max(general_heatmap_3c) * 255
            general_heatmap_3c = np.uint8(general_heatmap_3c)

            # print("img shape", img.shape)
            # print("general heatmap 3c shape", general_heatmap_3c.shape)

            img_with_heatmaps = cv2.addWeighted(img, 0.35, general_heatmap_3c, 0.65, 0)

            # img_with_heatmaps = T.CenterCrop(size=(prepad_h, prepad_w))(Image.fromarray(img_with_heatmaps))

            # img_with_heatmaps = cv2.resize(
            #     np.array(img_with_heata
            # general_heatmap = cv2.resize(
            #     general_heatmap,
            #     (orig_w, orig_h),
            #     interpolation=cv2.INTER_CUBIC
            # )

            # img_with_heatmaps = cv2.warpAffine(
            #     img_with_heatmaps,
            #     inv_trans, (int(orig_w), int(orig_h)),
            #     flags=cv2.INTER_LINEAR
            # )

            # mmcv.image.imwrite(general_heatmap, out_general_heatmap_orig)
            mmcv.image.imwrite(img_with_heatmaps, out_general_heatmap_file)
            # mmcv.image.imwrite(
            #     input_img,
            #     os.path.join(
            #         args.out_img_root,
            #         "heatmaps",
            #         f'vis_{osp.splitext(osp.basename(image_name))[0]}_orig.jpg'
            #     )
            # )


    if args.out_txt_root != '':
        os.makedirs(args.out_txt_root, exist_ok=True)
        old_filename = osp.splitext(osp.basename(args.json_file))[0]
        new_filename = old_filename.replace("COCO_anns", "poses")
        out_txt_path = osp.join(args.out_txt_root, "{}.csv".format(new_filename))
        estimated_poses.to_csv(out_txt_path, index=False, float_format="%.4f")


if __name__ == '__main__':
    args = parse_args()
    main(args)
