# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import warnings
from argparse import ArgumentParser
from mmcv.image import imwrite
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
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
    parser.add_argument(
        '--output-json',
        action="store_true",
        default=False,
        help='whether to save poses as json')
    parser.add_argument(
        '--output-heatmap',
        action="store_true",
        default=False,
        help='whether to save heatmaps as images')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    return args


def main(args):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    assert args.show or (args.out_img_root != '')
    assert args.img != '' or args.img_root != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
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

    image_name = os.path.join(args.img_root, args.img)
    print("Image name -", image_name)
    if os.path.isdir(image_name):
        print("Running the mmpose on the whole folder")
        images_names = list(map(
            lambda x: os.path.join(image_name, x),
            [dr for dr in os.listdir(image_name) if os.path.isfile(
                os.path.join(image_name, dr)
            )]
        ))
    else:
        print("Running the mmpose on a single image")
        images_names = [image_name]

    if args.output_json:
        json_dict = dict()

    for img_i, image_name in enumerate(images_names):
        _, relative_image_name = os.path.split(image_name)
        
        if args.output_json:
            json_dict[relative_image_name] = []

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = True

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        try:
            heatmap = np.mean(returned_outputs[0]['heatmap'], axis=0)
        except IndexError:
            continue
        # print(heatmap.shape)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                "vis_{:s}".format(relative_image_name)
            )
            out_general_heat_file = os.path.join(
                args.out_img_root,
                "vis_heatmap_{:s}".format(relative_image_name)
            )
            out_heat_folder = os.path.join(
                args.out_img_root,
                "heatmaps",
                relative_image_name.split(".")[0]
            )

        # print("----------")
        # print(image_name)
        
        # show the results
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

        if args.output_heatmap:
            
            general_heatmap = np.max(heatmap, axis=0)
            general_heatmap = ((general_heatmap - np.min(general_heatmap[:])) * 255)
            # imwrite(general_heatmap, out_general_heat_file)

            for kpt_channel in range(heatmap.shape[0]):
                htm = heatmap[kpt_channel, :, :].squeeze()
                htm = ((htm - np.min(htm[:])) * 255)
                # print("{}, {:.2f} - {:.2f}".format(htm.shape, np.min(htm), np.max(htm)))
                imwrite(
                    htm,
                    os.path.join(
                        out_heat_folder,
                        "kpt_{}.{}".format(
                            dataset_info.keypoint_info[kpt_channel]["name"],
                            relative_image_name.split(".")[-1])
                    )
                )

        if args.output_json:
            for pose in pose_results:
                json_dict[relative_image_name].append(pose['keypoints'].tolist())

        print("\r{:d}/{:d} ({:.2f}%)".format(
            img_i+1,
            len(images_names),
            (img_i+1)/len(images_names)*100
        ), end="")
    print()

    if args.output_json:
        print("Saving the JSON output")

        if args.out_img_root:
            save_filename = os.path.join(args.out_img_root, "estimated_2d_poses.json")
        else:
            save_filename = os.path.join(args.img_root, "estimated_2d_poses.json")

        with open(save_filename, "w") as fp:
            json.dump(
                json_dict,
                fp,
                indent=2
            )

if __name__ == '__main__':
    args = parse_args()
    main(args)
