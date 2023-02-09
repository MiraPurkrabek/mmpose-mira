# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from argparse import ArgumentParser
import cv2
import numpy as np

import mmcv

from mmpose.apis import (inference_bottom_up_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)

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
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img-root',
        type=str,
        help='Path to an image file or a image folder.')
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
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--pose-nms-thr',
        type=float,
        default=0.9,
        help='OKS threshold for pose NMS')
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

    assert args.show or (args.out_img_root != '')

    # prepare image list
    if osp.isfile(args.img_root):
        image_list = [args.img_root]
    elif osp.isdir(args.img_root):
        image_list = [
            osp.join(args.img_root, fn) for fn in os.listdir(args.img_root)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))
        ]
    else:
        raise ValueError('Image path should be an image or image folder.'
                         f'Got invalid image path: {args.img_root}')

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
        assert (dataset == 'BottomUpCocoDataset')
    else:
        dataset_info = DatasetInfo(dataset_info)

    # optional
    return_heatmap = hasattr(args, "output_heatmap") and args.output_heatmap

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # process each image
    for image_name in mmcv.track_iter_progress(image_list):

        # test a single image, with a list of bboxes.
        print()
        print(image_name)
        pose_results, returned_outputs = inference_bottom_up_pose_model(
            pose_model,
            image_name,
            dataset=dataset,
            dataset_info=dataset_info,
            pose_nms_thr=args.pose_nms_thr,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(
                args.out_img_root,
                "keypoints",
                f'vis_{osp.splitext(osp.basename(image_name))[0]}.jpg')
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

                sep_heatmaps_dir = os.path.join(
                    args.out_img_root,
                    "sep_heatmaps"
                )
                os.makedirs(sep_heatmaps_dir, exist_ok=True)


        # show the results
        vis_pose_result(
            pose_model,
            image_name,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=args.show,
            out_file=out_file)

        if return_heatmap:
            heatmap = np.mean(returned_outputs[0]['heatmap'], axis=0)
            general_heatmap = np.max(heatmap, axis=0)
            general_heatmap = ((general_heatmap - np.min(general_heatmap[:])) * 255)
            img = mmcv.image.imread(image_name)

            orig_h, orig_w, _ = img.shape

            general_heatmap_3c = np.zeros((orig_h, orig_w, 3))
            
            for kpt_channel in range(1, heatmap.shape[0]):
                htm = heatmap[kpt_channel, :, :].squeeze()
                htm = ((htm - np.min(htm[:])) * 255)
                big_htm = cv2.resize(
                    htm,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_CUBIC,
                )

                colored_sep_htmp = cv2.applyColorMap(
                    big_htm.astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                colored_sep_htmp = cv2.addWeighted(img, 0.35, colored_sep_htmp, 0.65, 0)
                save_path = os.path.join(
                    sep_heatmaps_dir,
                    "vis_{:s}_heatmap_{:s}.jpg".format(
                        osp.splitext(osp.basename(image_name))[0],
                        keypoints_names_short[kpt_channel-1],
                    ),
                )
                mmcv.image.imwrite(
                    colored_sep_htmp,
                    save_path,
                )

                
                general_heatmap_3c[:, :, int(kpt_channel%3)] += big_htm

                color = [100, 100, 100]
                color[int(kpt_channel%3)] = 255
                coors = np.unravel_index(big_htm.argmax(), big_htm.shape)

                peak = big_htm[coors]

                # Only show those keypoints with ppt higher than 10%
                if peak/255 > 0.1:

                    general_heatmap_3c = cv2.putText(
                        general_heatmap_3c,
                        "{:s} ({:.2f})".format(
                            keypoints_names_short[kpt_channel-1],
                            peak/255
                        ),
                        coors[::-1],
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1,
                        color = color,
                        thickness = 2,
                    )

            
            general_heatmap_3c = general_heatmap_3c / np.max(general_heatmap_3c) * 255
            general_heatmap_3c = np.uint8(general_heatmap_3c)

            img_with_heatmaps = cv2.addWeighted(img, 0.35, general_heatmap_3c, 0.65, 0)

            mmcv.image.imwrite(img_with_heatmaps, out_general_heatmap_file)


if __name__ == '__main__':
    main(parse_args())
