import cv2
from scipy.ndimage import rotate
import numpy as np

RANDOM_GENERATOR = np.random.default_rng()

def random_choice(array, size=1):
    ch = RANDOM_GENERATOR.choice(array, size=size)
    if size == 1:
        ch = ch[0]
    return ch

def random_float(a_min=0, a_max=1):
    rnd = RANDOM_GENERATOR.random()
    return (a_max - a_min) * rnd + a_min

def random_int(a_min=0, a_max=1):
    return int(random_float(a_min=a_min, a_max=a_max))

def get_random_bbox(img_shape, min_size):
    img_h, img_w = img_shape[:2]

    top = random_int(a_max=(1-min_size) * img_h)
    min_bottom = top + int(min_size*img_h)
    bottom = min_bottom + random_int(a_max=(img_h-min_bottom))
    
    left = random_int(a_max=(1-min_size) * img_w)
    min_right = left + int(min_size*img_w)
    right = min_right + random_int(a_max=(img_w-min_right))

    assert right <= img_w, "{}<={}".format(right, img_w)
    assert left <= img_w, "{}<={}".format(left, img_w)
    assert top <= img_h, "{}<={}".format(top, img_h)
    assert bottom <= img_h, "{}<={}".format(bottom, img_h)
    
    return (left, top, right, bottom)

def get_random_angle(min_angle, angle_type="degree"):
    if angle_type.lower() == "degree":
        max_possible_angle = 360
    elif angle_type.lower() == "radian":
        max_possible_angle = 2 * np.pi
    else:
        raise ValueError("Unknown angle type")
    
    assert min_angle >= 0
    assert min_angle <= max_possible_angle

    random_angle = random_float(
        a_min = min_angle,
        a_max = max_possible_angle
    )

    if angle_type.lower() == "degree":
        random_angle = int(random_angle)

    return random_angle

def random_crop(img, min_size=0.5):
    random_bbox = get_random_bbox(img.shape, min_size=min_size)
    return img[random_bbox[1]:random_bbox[3], random_bbox[0]:random_bbox[2], :]

def gradual_crop(img, direction="horizontal", n_steps=10, reverse=False):
    img_h, img_w = img.shape[:2]

    assert direction.lower() in [
        "horizontal",
        "vertical",
        "both",
    ]

    for s in np.linspace(0, 1, n_steps, endpoint=False):
        if reverse:
            left = 0
            top = 0
            right = img_w if direction.lower() == "vertical" else int((1-s) * img_w)
            bottom = img_h if direction.lower() == "horizontal" else int((1-s) * img_h)
        else:
            left = 0 if direction.lower() == "vertical" else int(s * img_w)
            top = 0 if direction.lower() == "horizontal" else int(s * img_h)
            right = img_w
            bottom = img_h
        
        ret_img = img.copy()
        yield ret_img[top:bottom, left:right, :]

def random_occlude(img, min_size=0.1, occ_type="black"):

    assert occ_type.lower() in [
        "black",
        "white",
        "random",
    ]

    random_bbox = get_random_bbox(img.shape, min_size=min_size)

    if occ_type.lower() == "black":
        patch = np.zeros((random_bbox[3]-random_bbox[1], random_bbox[2]-random_bbox[0], 3))
    elif occ_type.lower() == "white":
        patch = 255 * np.ones((random_bbox[3]-random_bbox[1], random_bbox[2]-random_bbox[0], 3))
    elif occ_type.lower() == "random":
        patch = 255 * RANDOM_GENERATOR.random((random_bbox[3]-random_bbox[1], random_bbox[2]-random_bbox[0], 3))
    
    img[random_bbox[1]:random_bbox[3], random_bbox[0]:random_bbox[2], :] = patch
    return img

def gradual_occlude(img, direction="horizontal", n_steps=10, reverse=False, occ_type="black", startpoint=False):
    img_h, img_w = img.shape[:2]

    assert direction.lower() in [
        "horizontal",
        "vertical",
        "both",
    ]

    assert occ_type.lower() in [
        "black",
        "white",
        "random",
    ]

    for s in np.linspace(1, 0, n_steps, endpoint=startpoint)[::-1]:
        if reverse:
            left = 0
            top = 0
            right = img_w if direction.lower() == "vertical" else int(s * img_w)
            bottom = img_h if direction.lower() == "horizontal" else int(s * img_h)
        else:
            left = 0 if direction.lower() == "vertical" else int((1-s) * img_w)
            top = 0 if direction.lower() == "horizontal" else int((1-s) * img_h)
            right = img_w
            bottom = img_h

        if occ_type.lower() == "black":
            patch = np.zeros((bottom-top, right-left, 3))
        elif occ_type.lower() == "white":
            patch = 255 * np.ones((bottom-top, right-left, 3))
        elif occ_type.lower() == "random":
            patch = 255 * RANDOM_GENERATOR.random((bottom-top, right-left, 3))
        
        ret_img = img.copy()
        ret_img[top:bottom, left:right, :] = patch
        yield ret_img

def random_rotate(img, min_angle_deg=5):
    return rotate(img, get_random_angle(min_angle=min_angle_deg), reshape=True)

def gradual_rotate(img, n_steps=10):
    for angle_deg in np.linspace(0, 360, n_steps, endpoint=False):
        yield  rotate(img.copy(), angle_deg, reshape=True)

def mirror(img):
    return img[:, ::-1, :]

def gradual_rescale(img, n_steps=10, min_area=5000):
    img_h, img_w = img.shape[:2]
    
    min_scale = np.sqrt(min_area / (img_w * img_h))
    min_w = min_scale * img_w
    min_h = min_scale * img_h

    for w, h in zip(np.linspace(img_w, min_w, n_steps), np.linspace(img_h, min_h, n_steps)):
    
        rescaled_img = cv2.resize(
            img,
            (int(w), int(h)),
            interpolation = cv2.INTER_AREA,
        )

        yield rescaled_img

def gradual_blur(img, blur_type="Gaussian", n_steps=10):

    assert blur_type.lower() in [
        "gaussian",
        "motion_v",
        "motion_h",
        "motion_vh",
    ]

    for blur_size in np.linspace(5, 501, n_steps, endpoint=True):
        blur_size = int(blur_size)

        if blur_type.lower() == "gaussian":
            blurred = cv2.blur(
                img,
                (blur_size, blur_size),
                borderType = cv2.BORDER_REFLECT,
            )

        elif blur_type.lower().startswith("motion_"):
            kernel_v = np.zeros((blur_size, blur_size))
            kernel_v[:, int((blur_size - 1)/2)] = np.ones(blur_size)
            
            kernel_h = np.zeros((blur_size, blur_size))
            kernel_h[int((blur_size - 1)/2), :] = np.ones(blur_size)

            if blur_type.lower().endswith("_v"):
                blurred = cv2.filter2D(img, -1, kernel_v)
            elif blur_type.lower().endswith("_h"):
                blurred = cv2.filter2D(img, -1, kernel_h)
            elif blur_type.lower().endswith("_vh"):
                blurred = cv2.filter2D(img, -1, kernel_v)
                blurred = cv2.filter2D(blurred, -1, kernel_h)
            
        yield blurred

def gradual_noise(img, noise_type="SaltAndPepper", n_steps=10):
    
    assert noise_type.lower() in [
        "gaussian",
        "saltandpepper",
    ]
    row, col, ch = img.shape

    for noise_size in np.linspace(0, 1, n_steps, endpoint=True):
        
        if noise_type.lower() == "gaussian":

            mean = 0
            var = noise_size
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch)) * 255
            gauss = gauss.reshape(row,col,ch)
            noised = img + gauss
            noised = np.clip(noised, 0, 255)
            
        elif noise_type.lower() == "saltandpepper":
            s_vs_p = 0.5
            amount = noise_size
            noised = np.copy(img)
            
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in img.shape[:2]]
            noised[coords[0], coords[1], :] = 0

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in img.shape[:2]]
            noised[coords[0], coords[1], :] = 255
            
        yield noised

    return img

