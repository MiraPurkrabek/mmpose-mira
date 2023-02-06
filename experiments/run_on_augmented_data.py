import cv2
import os
from scipy.ndimage import rotate
import shutil
from tqdm import tqdm
import json
import warnings
from argparse import ArgumentParser
import time
import numpy as np

from functools import partial

from augmentations import (
    random_choice,
    random_float,
    random_crop,
    gradual_crop,
    random_occlude,
    gradual_occlude,
    random_rotate,
    gradual_rotate,
    mirror,
    gradual_rescale,
    gradual_blur,
    gradual_noise,
)

from demo.top_down_img_demo import main, parse_args

class bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)

FOLDER = "/datagrid/personal/purkrmir/data/pose_experiments/running"

# AUG_TYPE = "NONE"           # Do not perform any augmentations
# AUG_TYPE = "RECURSIVE"      # Execute all augmentations in the given order, each for each image
AUG_TYPE = "CONSECUTIVE"    # Execute all augmentations in the given order
# AUG_TYPE = "RANDOM"         # Randomly make 4 agmentations from the list

# AUG_FOLDER_NAME = "not_augmented"
# AUG_FOLDER_NAME = "recursive"
AUG_FOLDER_NAME = "consecutive"
# AUG_FOLDER_NAME = "augmented"

N_STEPS=7
N_IMAGES=2                # Applies only for 'RANDOM' AUG_TYPE

# POSE_TYPE = "HRNet"
POSE_TYPE = "SWIN"

POSSIBLE_EDITS = [
    # bind(random_crop),
    # bind(random_occlude, occ_type="random"),
    # bind(random_occlude, occ_type="white"),
    # bind(random_occlude, occ_type="black"),
    # bind(random_rotate),
    # bind(mirror),
    
    bind(gradual_occlude, direction="vertical", reverse=True, occ_type="random", n_steps=N_STEPS),
    bind(gradual_occlude, direction="horizontal", reverse=True, occ_type="random", n_steps=N_STEPS),
    bind(gradual_occlude, direction="both", reverse=True, occ_type="random", n_steps=N_STEPS),
    
    # bind(gradual_rotate, n_steps=N_STEPS),
    # bind(gradual_rescale, n_steps=N_STEPS),
    # bind(gradual_blur),
    # bind(gradual_noise, noise_type="SaltAndPepper", n_steps=N_STEPS),
    # bind(gradual_noise, noise_type="Gaussian", n_steps=N_STEPS),
]

###############################################################################

def save_augmented_img(img, save_name, aug_folder, json_out):
    cv2.imwrite(
        os.path.join(aug_folder, "{:s}".format(save_name)),
        img
    )

    random_id = int(random_float() * 109238129071281938)
    json_out["images"].append({
        "file_name": "{:s}/{:s}".format(AUG_FOLDER_NAME, save_name),
        "id": random_id
    })
    json_out["annotations"].append({
        "image_id": random_id,
        "bbox": [0, 0, img.shape[1], img.shape[0]],
        "score": 0.99,
        "id": int(random_float() * 109238129071281938)
    })

    return json_out

def recursive_image_augmentation(img, possible_edits, save_name, save_ext, aug_folder, json_out, progress_bar):
    if len(possible_edits) == 0:
        progress_bar.update()
        return save_augmented_img(img, "{:s}.{:s}".format(save_name, save_ext), aug_folder, json_out)

    edit_fcn = possible_edits[0]

    for i, aug_img in enumerate(edit_fcn(img)):
        json_out = recursive_image_augmentation(
            aug_img,
            possible_edits[1:],
            save_name + "_{:03d}".format(i),
            save_ext,
            aug_folder,
            json_out,
            progress_bar,
        )

    return json_out

def prepare_images(folder, json_filepath, n_images=100, max_edits=4):
    
    # ToDo: Also take JPEG and PNG
    images = [f for f in os.listdir(folder) if f.endswith(".jpg")]

    aug_folder = os.path.join(folder, AUG_FOLDER_NAME)
    os.makedirs(aug_folder, exist_ok=True)

    # Remove in this way to prevent removing already estimated poses
    for f in os.listdir(aug_folder):
        if os.path.isfile(os.path.join(aug_folder, f)):
            os.remove(os.path.join(aug_folder, f))

    json_out = {
        "info": {
            "description": "Data augmented by agressive agmentation",
            "year": 2023,
            "date_created": time.strftime('%Y/%m/%d', time.localtime())
        },
        "images": [],
        "annotations": []
    }

    if AUG_TYPE.upper() == "RANDOM":
            
        for i in tqdm(range(n_images), ascii=True):

            # Select random image from 'images'
            selected_img = random_choice(images)
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            img = cv2.imread(selected_img_path)
            
            save_name = "{:s}_random_edits_{:04d}.{:s}".format(
                img_name,
                i,
                "jpg"
            )

            for _ in range(max_edits):
                edit_function = random_choice(POSSIBLE_EDITS)
                img = edit_function(img)

            json_out = save_augmented_img(img, save_name, aug_folder, json_out)

    elif AUG_TYPE.upper() == "RECURSIVE":
        
        with tqdm(total= len(images) * (N_STEPS ** len(POSSIBLE_EDITS)), ascii=True) as progress_bar:

            for selected_img in images:
            
                img_name = ".".join(selected_img.split(".")[:-1])
                selected_img_path = os.path.join(folder, selected_img)
                img = cv2.imread(selected_img_path)
            
                recursive_image_augmentation(
                    img,
                    POSSIBLE_EDITS,
                    img_name,
                    "jpg",
                    aug_folder,
                    json_out,
                    progress_bar
                )
    elif AUG_TYPE.upper() == "CONSECUTIVE":
        
        with tqdm(total=len(images) * N_STEPS * len(POSSIBLE_EDITS), ascii=True) as progress_bar:

            idx = 0
            for selected_img in images:
            
                img_name = ".".join(selected_img.split(".")[:-1])
                selected_img_path = os.path.join(folder, selected_img)
                img = cv2.imread(selected_img_path)
            
                for edit_fcn in POSSIBLE_EDITS:
                    for aug_img in edit_fcn(img):
                        save_name = "{:s}_{:04d}.{:s}".format(
                            img_name,
                            idx,
                            "jpg"
                        )
                        json_out = save_augmented_img(aug_img, save_name, aug_folder, json_out)

                        idx += 1
                        progress_bar.update()


    with open(json_filepath, "w") as f:
        json.dump(json_out, f, indent = 2)
            

if __name__ == '__main__':

    if POSE_TYPE == "HRNet":
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth"
    else:
        POSE_CFG="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/swin_l_p4_w7_coco_384x288.py"
        POSE_PTH="https://download.openmmlab.com/mmpose/top_down/swin/swin_l_p4_w7_coco_384x288-c36b7845_20220705.pth"

    try:
        args = parse_args()
    except:
        print("Using default POSE config and weights")
        args = ArgumentParser().parse_args()

        json_filepath = os.path.join(
            FOLDER, "test.json"
        )
        AUG_FOLDER = os.path.join(FOLDER, AUG_FOLDER_NAME)

        args.pose_config = POSE_CFG
        args.pose_checkpoint = POSE_PTH
        args.img_root = FOLDER
        args.json_file = json_filepath
        args.show = False
        args.out_img_root = os.path.join(
            AUG_FOLDER,
            "out",
            "POSE_{:s}".format(POSE_TYPE),
        )
        args.out_txt_root = ""
        args.device = "cuda:0"
        args.kpt_thr = 0.3
        args.radius = 4
        args.thickness = 1
        args.output_heatmap = True

    prepare_images(FOLDER, json_filepath, N_IMAGES)

    main(args)
