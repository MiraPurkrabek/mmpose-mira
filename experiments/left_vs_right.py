import cv2
import os
from scipy.ndimage import rotate
import shutil
from tqdm import tqdm
import json
import warnings
from argparse import ArgumentParser
from random import choice, random
import time
import numpy as np

from demo.top_down_img_demo import main, parse_args

AUG_TYPE = "NONE"
# AUG_TYPE = "ROTATION"
# AUG_TYPE = "RANDOM"
# AUG_TYPE = "LOWRES"
# AUG_TYPE = "GAUSSBLUR"
# AUG_TYPE = "MOTIONBLUR"
# AUG_TYPE = "LOWRESMOTIONBLUR"
# AUG_TYPE = "GAUSSNOISE"
# AUG_TYPE = "SAPNOISE"
# AUG_TYPE = "LOWRESSAPNOISE"

N_IMAGES=600 # Applies only for RANDOM AUG_TYPE
FOLDER = "/datagrid/personal/purkrmir/data/pose_experiments/running"

# POSE_TYPE = "HRNet"
POSE_TYPE = "SWIN"


###############################################################################
if AUG_TYPE == "NONE":
    AUG_FOLDER_NAME = "not_augmented"
elif AUG_TYPE == "ROTATION":
    AUG_FOLDER_NAME = "rotated"
elif AUG_TYPE == "RANDOM":
    AUG_FOLDER_NAME = "augmented"
elif AUG_TYPE == "LOWRES":
    AUG_FOLDER_NAME = "low-res"
elif AUG_TYPE == "GAUSSBLUR":
    AUG_FOLDER_NAME = "GaussBlur"
elif AUG_TYPE == "MOTIONBLUR":
    AUG_FOLDER_NAME = "MotionBlur"
elif AUG_TYPE == "LOWRESMOTIONBLUR":
    AUG_FOLDER_NAME = "low-res_MotionBlur"
elif AUG_TYPE == "GAUSSNOISE":
    AUG_FOLDER_NAME = "GaussNoise"
elif AUG_TYPE == "SAPNOISE":
    AUG_FOLDER_NAME = "SaPNoise"
elif AUG_TYPE == "LOWRESSAPNOISE":
    AUG_FOLDER_NAME = "low-res_SaPNoise"
else:
    raise ValueError("Unknown AUGMENTATION_TYPE")

def random_crop(img, min_size=0.5):
    h, w, _ = img.shape

    min_h = int(random() * (1-min_size) * h)
    max_h = min_h + int(min_size*h) + int(random()*(h-min_h)*(1-min_size))
    
    min_w = int(random() * (1-min_size) * w)
    max_w = min_w + int(min_size*w) + int(random()*(w-min_w)*(1-min_size))
    
    assert (max_h - min_h)/h + 0.02  >= min_size, "{:d}:{:d}, {:d}".format(min_h, max_h, h)
    assert (max_w - min_w)/w + 0.02  >= min_size, "{:d}:{:d}, {:d}".format(min_w, max_w, w)

    return img[min_h:max_h, min_w:max_w, :]

def random_occlude(img, min_size=0.1):
    h, w, _ = img.shape
    
    min_h = int(random() * (1-min_size) * h)
    max_h = min_h + int(min_size*h) + int(random()*(h-min_h)*(1-min_size))
    
    min_w = int(random() * (1-min_size) * w)
    max_w = min_w + int(min_size*w) + int(random()*(w-min_w)*(1-min_size))

    assert (max_h - min_h)/h + 0.02 >= min_size, "{:d}:{:d}, {:d}".format(min_h, max_h, h)
    assert (max_w - min_w)/w + 0.02  >= min_size, "{:d}:{:d}, {:d}".format(min_w, max_w, w)
    
    img[min_h:max_h, min_w:max_w, :] = 0
    return img

def random_rotate(img, min_angle_deg=5):
    angle_deg = 0

    while abs(angle_deg) < min_angle_deg:
        angle_deg = random() * 360 - 180
    return rotate(img, angle_deg, reshape=True)

def mirror(img):
    return img[:, ::-1, :]

def augment_images(folder, json_filepath, n_images=100, max_edits=4):
    
    images = [f for f in os.listdir(folder) if f.endswith(".jpg")]

    aug_folder = os.path.join(folder, AUG_FOLDER_NAME)
    
    os.makedirs(aug_folder, exist_ok=True)

    # Remove in this way to prevent removing already estimated poses
    for f in os.listdir(aug_folder):
        if os.path.isfile(os.path.join(aug_folder, f)):
            os.remove(os.path.join(aug_folder, f))

    possible_edits = [
        # random_crop,
        random_rotate,
        random_occlude,
        mirror
    ]

    json_out = {
        "info": {
            "description": "Data augmented by agressive agmentation",
            "year": 2023,
            "date_created": time.strftime('%Y/%m/%d', time.localtime())
        },
        "images": [],
        "annotations": []
    }

    if AUG_TYPE == "RANDOM":
        for i in tqdm(range(n_images), ascii=True):
            
            # Select random image from 'images'
            selected_img = choice(images)
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            img = cv2.imread(selected_img_path)

            # Augment the image
            number_of_augmentations = int(random() * max_edits)
            for _ in range(number_of_augmentations):
                edit = choice(possible_edits)

                try:
                    img = edit(img)
                except AssertionError:
                    print("Skipping edit because of assertion error")

            cv2.imwrite(
                os.path.join(aug_folder, "{:s}_{:d}.jpg".format(img_name, i)),
                img
            )

            random_id = int(random() * 109238129071281938)
            json_out["images"].append({
                "file_name": "{:s}/{:s}_{:d}.jpg".format(AUG_FOLDER_NAME, img_name, i),
                "id": random_id
            })
            json_out["annotations"].append({
                "image_id": random_id,
                "bbox": [0, 0, img.shape[1], img.shape[0]],
                "score": 0.99,
                "id": "augmented_{:d}".format(i)
            })
    elif AUG_TYPE == "ROTATION":
        for i in tqdm(range(len(images))):

            for angle_deg in range(0, 360, 10):

                selected_img = images[i]
                img_name = ".".join(selected_img.split(".")[:-1])
                selected_img_path = os.path.join(folder, selected_img)
                img = cv2.imread(selected_img_path)


                while img.shape[0] * img.shape[1] < 40000:
                    img = cv2.resize(
                        img,
                        (int(img.shape[1]*2), int(img.shape[0]*2)),
                        interpolation = cv2.INTER_AREA,
                    )

                img = rotate(img, angle_deg, reshape=True)

                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_{:03d}.jpg".format(img_name, angle_deg)),
                    img
                )

                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, angle_deg),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, angle_deg)
                })
    elif AUG_TYPE == "NONE":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            img = cv2.imread(selected_img_path)


            while img.shape[0] * img.shape[1] < 40000:
                img = cv2.resize(
                    img,
                    (int(img.shape[1]*2), int(img.shape[0]*2)),
                    interpolation = cv2.INTER_AREA,
                )


            cv2.imwrite(
                os.path.join(aug_folder, "{:s}.jpg".format(img_name)),
                img
            )

            random_id = int(random() * 109238129071281938)
            json_out["images"].append({
                "file_name": "{:s}/{:s}.jpg".format(AUG_FOLDER_NAME, img_name),
                "id": random_id
            })
            json_out["annotations"].append({
                "image_id": random_id,
                "bbox": [0, 0, img.shape[1], img.shape[0]],
                "score": 0.99,
                "id": "augmented_{:d}".format(i)
            })
    elif AUG_TYPE == "LOWRES":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            s = 1

            while True:
                img = cv2.resize(
                    orig_img,
                    (int(orig_img.shape[1]*s), int(orig_img.shape[0]*s)),
                    interpolation = cv2.INTER_AREA,
                )

                if img.shape[0] * img.shape[1] < 5000:
                    break

                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_scale_{:03d}.jpg".format(img_name, int(s*100))),
                    img
                )

                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_scale_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, int(s*100)),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, int(s*100))
                })

                s /= 1.5
    elif AUG_TYPE == "GAUSSBLUR":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for b in range(10, 500, 50):
                img = cv2.blur(
                    orig_img,
                    (b, b),
                    borderType = cv2.BORDER_REFLECT,
                )

                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_blur_{:03d}.jpg".format(img_name, b)),
                    img
                )

                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_blur_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, b),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, b)
                })
    elif AUG_TYPE == "MOTIONBLUR":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for b in range(10, 500, 50):
  
                kernel_v = np.zeros((b, b))
                kernel_h = np.zeros((b, b))
                kernel_v[:, int((b - 1)/2)] = np.ones(b)
                kernel_h[int((b - 1)/2), :] = np.ones(b)
                kernel_v /= b
                kernel_h /= b

                img_v = cv2.filter2D(orig_img, -1, kernel_v)
                img_h = cv2.filter2D(orig_img, -1, kernel_h)
                img_vh = cv2.filter2D(img_v, -1, kernel_h)

                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_vblur_{:03d}.jpg".format(img_name, b)),
                    img_v
                )
                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_hblur_{:03d}.jpg".format(img_name, b)),
                    img_h
                )
                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_vhblur_{:03d}.jpg".format(img_name, b)),
                    img_vh
                )

                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_vblur_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, b),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img_v.shape[1], img_v.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, b)
                })
                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_hblur_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, b),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img_h.shape[1], img_h.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, b)
                })
                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_vhblur_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, b),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img_vh.shape[1], img_vh.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, b)
                })
    elif AUG_TYPE == "LOWRESMOTIONBLUR":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for b in range(10, 500, 100):
  
                kernel_v = np.zeros((b, b))
                kernel_h = np.zeros((b, b))
                kernel_v[:, int((b - 1)/2)] = np.ones(b)
                kernel_h[int((b - 1)/2), :] = np.ones(b)
                kernel_v /= b
                kernel_h /= b

                img_v = cv2.filter2D(orig_img, -1, kernel_v)
                img_h = cv2.filter2D(orig_img, -1, kernel_h)
                img_vh = cv2.filter2D(img_v, -1, kernel_h)

                imgs = [img_v, img_h, img_vh]
                img_names = [
                    "{:s}_vblur_{:03d}".format(img_name, b),
                    "{:s}_hblur_{:03d}".format(img_name, b),
                    "{:s}_vhblur_{:03d}".format(img_name, b),
                ]

                for img, img_n in zip(imgs, img_names):

                    s = 1.0
                    while True:
    
                        small_img = cv2.resize(
                            img,
                            (int(img.shape[1]*s), int(img.shape[0]*s))
                        )

                        if small_img.shape[0] * small_img.shape[1] < 5000:
                            break

                        save_name = "{:s}_scale_{:03d}.jpg".format(img_n, int(s*100))

                        cv2.imwrite(
                            os.path.join(aug_folder, save_name),
                            small_img,
                        )

                        random_id = int(random() * 109238129071281938)
                        json_out["images"].append({
                            "file_name": "{:s}/{:s}".format(AUG_FOLDER_NAME, save_name),
                            "id": random_id
                        })
                        json_out["annotations"].append({
                            "image_id": random_id,
                            "bbox": [0, 0, small_img.shape[1], small_img.shape[0]],
                            "score": 0.99,
                            "id": "augmented_{:d}_{:03d}_{:03d}".format(i, b, int(s*100))
                        })

                        s /= 1.5
    elif AUG_TYPE == "GAUSSNOISE":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for n in range(1, 200, 10):
  
                row,col,ch= orig_img.shape
                mean = 0
                var = n / 100
                sigma = var**0.5
                gauss = np.random.normal(mean,sigma,(row,col,ch)) * 255
                gauss = gauss.reshape(row,col,ch)
                img = orig_img + gauss
                img = np.clip(img, 0, 255)

                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_noise_{:03d}.jpg".format(img_name, n)),
                    img
                )
                
                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_noise_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, n),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, n)
                })
    elif AUG_TYPE == "SAPNOISE":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for n in range(1, 1000, 50):
  
                row,col,ch= orig_img.shape
                s_vs_p = 0.5
                amount = n / 1000
                img = np.copy(orig_img)
                # Salt mode
                num_salt = np.ceil(amount * img.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt))
                        for i in orig_img.shape[:2]]
                img[coords[0], coords[1], :] = 0

                # Pepper mode
                num_pepper = np.ceil(amount* orig_img.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper))
                        for i in orig_img.shape[:2]]
                img[coords[0], coords[1], :] = 255
                
                cv2.imwrite(
                    os.path.join(aug_folder, "{:s}_sapnoise_{:03d}.jpg".format(img_name, n)),
                    img
                )
                
                random_id = int(random() * 109238129071281938)
                json_out["images"].append({
                    "file_name": "{:s}/{:s}_sapnoise_{:03d}.jpg".format(AUG_FOLDER_NAME, img_name, n),
                    "id": random_id
                })
                json_out["annotations"].append({
                    "image_id": random_id,
                    "bbox": [0, 0, img.shape[1], img.shape[0]],
                    "score": 0.99,
                    "id": "augmented_{:d}_{:03d}".format(i, n)
                })
    elif AUG_TYPE == "LOWRESSAPNOISE":
        for i in tqdm(range(len(images))):

            selected_img = images[i]
            img_name = ".".join(selected_img.split(".")[:-1])
            selected_img_path = os.path.join(folder, selected_img)
            orig_img = cv2.imread(selected_img_path)

            for n in range(1, 1000, 150):
  
                row,col,ch= orig_img.shape
                s_vs_p = 0.5
                amount = n / 1000
                img = np.copy(orig_img)
                # Salt mode
                num_salt = np.ceil(amount * img.size * s_vs_p)
                coords = [np.random.randint(0, i - 1, int(num_salt))
                        for i in orig_img.shape[:2]]
                img[coords[0], coords[1], :] = 0

                # Pepper mode
                num_pepper = np.ceil(amount* orig_img.size * (1. - s_vs_p))
                coords = [np.random.randint(0, i - 1, int(num_pepper))
                        for i in orig_img.shape[:2]]
                img[coords[0], coords[1], :] = 0

                s = 1.0
                while True:

                    small_img = cv2.resize(
                        img,
                        (int(img.shape[1]*s), int(img.shape[0]*s))
                    )

                    if small_img.shape[0] * small_img.shape[1] < 5000:
                        break

                    save_name = "{:s}_sapnoise_{:03d}_scale_{:03d}.jpg".format(img_name, n, int(s*100))

                    cv2.imwrite(
                        os.path.join(aug_folder, save_name),
                        small_img,
                    )

                    random_id = int(random() * 109238129071281938)
                    json_out["images"].append({
                        "file_name": "{:s}/{:s}".format(AUG_FOLDER_NAME, save_name),
                        "id": random_id
                    })
                    json_out["annotations"].append({
                        "image_id": random_id,
                        "bbox": [0, 0, small_img.shape[1], small_img.shape[0]],
                        "score": 0.99,
                        "id": "augmented_{:d}_{:03d}_{:03d}".format(i, n, int(s*100))
                    })

                    s /= 1.5



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

    augment_images(FOLDER, json_filepath, N_IMAGES)

    main(args)
