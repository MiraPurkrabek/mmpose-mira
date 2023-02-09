# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules

SUPPORTED_FORMATS = ["jpg", "png", "jpeg"]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--show', default=False, help='Showing the results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmpose into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # Create array of images if the input is folder
    if os.path.isdir(args.img):
        args.img = [os.path.join(args.img, f) for f in os.listdir(args.img) if f.split(".")[-1].lower() in SUPPORTED_FORMATS]
    else:
        args.img = [args.img]

    # Prepare the output folder
    if args.out_file is None:
        root_dir = os.path.join(os.path.dirname(args.img[0]), "vis_pose")
    else:
        if os.path.exists(args.out_file) and os.path.isdir(args.out_file):
            root_dir = args.out_file
        else:
            raise ValueError("Unknown output location")
    args.out_file = [os.path.join(root_dir, "vis_"+os.path.basename(f)) for f in args.img]
    
    for img, out_file in zip(args.img, args.out_file):

        # inference a single image
        results = inference_topdown(model, img)
        results = merge_data_samples(results)

        # show the results
        img = imread(img, channel_order='rgb')
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=True,
            draw_heatmap=args.draw_heatmap,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
