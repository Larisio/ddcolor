import yaml
import glob
import cv2
import os
import numpy as np


def pull_image_paths(pull_paths, deep_pull, img_types):
    """
    Pulls image paths from the specified directory based on the given options.

    Args:
        pull_path (list): The paths to pull images from.
        deep_pull (bool): Whether to pull images from subdirectories.
        img_types (list): List of image file extensions to include.

    Returns:
        list: A list of image paths.
    """
    if deep_pull:
        img_paths = []
        for pull_path in pull_paths:
            print(f"[INFO] Pulling images from: {pull_path}")
            if not pull_path.endswith('/'):
                pull_path += '/'

            for img_type in img_types:
                print(f"[INFO] Searching for images of type: {img_type}")
                img_paths.extend(glob.glob(f"{pull_path}/**/*.{img_type}", recursive=True))

    else:
        for pull_path in pull_paths:
            print(f"[INFO] Pulling images from: {pull_path}")
            if not pull_path.endswith('/'):
                pull_path += '/'
            for img_type in img_types:
                print(f"[INFO] Searching for images of type: {img_type}")
                img_paths = img_paths.extend(glob.glob(f"{pull_path}/*.{img_type}"))

    return img_paths

def shuffle_image_paths(img_paths):
    """
    Shuffles the list of image paths.

    Args:
        img_paths (list): List of image paths to shuffle.

    Returns:
        list: Shuffled list of image paths.
    """
    import random
    random.shuffle(img_paths)
    return img_paths

def cut_and_split_image_paths(img_paths, max_images, split_ratio):
    """
    Splits the image paths into training and validation sets based on the given ratio.

    Args:
        img_paths (list): List of image paths to split.
        split_ratio (float): Ratio for splitting the dataset (e.g., 0.8 for 80% training, 20% validation).

    Returns:
        tuple: Two lists containing training and validation image paths.
    """
    # Cut
    img_paths = img_paths[:max_images] if max_images > 0 else img_paths

    split_index = int(len(img_paths) * split_ratio)
    train_paths = img_paths[:split_index]
    val_paths = img_paths[split_index:]
    return train_paths, val_paths

def filter_img_size(img, min_size, max_size):
    """
    Filters images based on their size.

    Args:
        img (numpy.ndarray): The image to check.
        min_size (int): Minimum size in pixels.
        max_size (int): Maximum size in pixels.

    Returns:
        bool: True if the image size is within the specified range, False otherwise.
    """
    height, width = img.shape[:2]
    if height < min_size or width < min_size:
        return True  # Skip images smaller than min_size
    if height > max_size or width > max_size:
        return True  # Skip images larger than max_size
    return False  # Image size is within the specified range

def filter_img_brightness(img, max_black_percentage, black_threshold, max_white_percentage, white_threshold):
    total_pixels = img.size

    # Zähle sehr dunkle und sehr helle Pixel
    num_dunkel = np.sum(img < black_threshold)
    num_hell = np.sum(img > white_threshold)

    anteil_dunkel = num_dunkel / total_pixels
    anteil_hell = num_hell / total_pixels


    if anteil_dunkel > max_black_percentage:
        return -1
    elif anteil_hell > max_white_percentage:
        return 1
    else:
        return 0
    
def cut_image_to_size(
    image,
    horizontal_cut=True,
    horizontal_cut_size=1024,
    horizontal_cut_direction='both',  # 'left', 'right', 'both'
    vertical_cut=True,
    vertical_cut_size=1024,
    vertical_cut_direction='top'      # 'top', 'bottom', 'both'
):
    """
    Cuts an image to specified width and height based on direction.

    Args:
        image (np.array): Input image (as loaded by OpenCV).
        horizontal_cut (bool): Whether to crop width.
        horizontal_cut_size (int): Target width in pixels.
        horizontal_cut_direction (str): 'left', 'right', or 'both'.
        vertical_cut (bool): Whether to crop height.
        vertical_cut_size (int): Target height in pixels.
        vertical_cut_direction (str): 'top', 'bottom', or 'both'.

    Returns:
        Cropped image as np.array.
    """

    h, w = image.shape[:2]

    # --- Horizontal Crop ---
    if horizontal_cut and w > horizontal_cut_size:
        if horizontal_cut_direction == 'left':
            x_start = 0
        elif horizontal_cut_direction == 'right':
            x_start = w - horizontal_cut_size
        elif horizontal_cut_direction == 'both':
            x_start = (w - horizontal_cut_size) // 2
        else:
            raise ValueError("Invalid horizontal_cut_direction")

        x_end = x_start + horizontal_cut_size
        image = image[:, x_start:x_end]

    # --- Vertical Crop ---
    if vertical_cut and h > vertical_cut_size:
        if vertical_cut_direction == 'top':
            y_start = 0
        elif vertical_cut_direction == 'bottom':
            y_start = h - vertical_cut_size
        elif vertical_cut_direction == 'both':
            y_start = (h - vertical_cut_size) // 2
        else:
            raise ValueError("Invalid vertical_cut_direction")

        y_end = y_start + vertical_cut_size
        image = image[y_start:y_end, :]

    return image

def resize_image(img, resize=True, resize_until=512):
    """
    Resizes the image proportionally so that neither width nor height
    exceeds `squeeze_until`.

    Args:
        img (numpy.ndarray): The input image.
        squeeze (bool): Whether to apply resizing.
        squeeze_until (int): Max allowed width or height.

    Returns:
        numpy.ndarray: Resized image if needed, otherwise original.
    """
    if not resize:
        return img

    h, w = img.shape[:2]

    # Only resize if any dimension exceeds the limit
    if w > resize_until or h > resize_until:
        scale = min(resize_until / w, resize_until / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img

def load_and_store_images(
        path_list,
        output_dir,
        max_images,
        txt_path_prefix,
        print_steps,

        #filter
        max_img_size,
        min_img_size,
        max_black_percentage,
        black_threshold,
        max_white_percentage,
        white_threshold,

        #cut
        horizontal_cut,
        horizontal_cut_size,
        horizontal_cut_direction,
        vertical_cut,
        vertical_cut_size,
        vertical_cut_direction,
        #squezze
        resize,
        resize_until,

        prefix='img_',
        verbose=True):
    """
    Loads images from the provided path list and stores them in the specified output directory.

    Args:
        path_list (list of str): List of image file paths.
        output_dir (str): Directory where images should be saved.
        rename (bool): Whether to rename images (True) or keep original names (False).
        prefix (str): Prefix to use for renamed images.
        verbose (bool): Whether to print loading/saving status.

    Returns:
        saved_paths (list of str): List of paths to the saved images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_paths = []
    touched_paths = []
    img_counter = 1
    for idx, path in enumerate(path_list):
        if img_counter > max_images:
            break

        image_name = os.path.basename(path)
        image_name = os.path.splitext(image_name)[0]

        img = cv2.imread(path)
        if img is None:
            if verbose:
                print(f"[Warning] Failed to load: {path}")
            continue

        # Filter
        if filter_img_size(img, min_img_size, max_img_size):
            if verbose:
                print(f"[Warning] Skipping image due to size: {path}")
            continue
        
        # Cutting
        img = cut_image_to_size(
            img,
            horizontal_cut=horizontal_cut,
            horizontal_cut_size=horizontal_cut_size,
            horizontal_cut_direction=horizontal_cut_direction,
            vertical_cut=vertical_cut,
            vertical_cut_size=vertical_cut_size,
            vertical_cut_direction=vertical_cut_direction
        )
        # Squeezing
        img = resize_image(img, resize=resize, resize_until=resize_until)

        # Brightness filter
        brightness_check = filter_img_brightness(img, max_black_percentage, black_threshold, max_white_percentage, white_threshold)
        if  brightness_check != 0:
            if verbose:
                if brightness_check == -1:
                    print(f"[Warning] Skipping image to dark: {path}")
                elif brightness_check == 1:
                    print(f"[Warning] Skipping image to brightness: {path}")
            continue

        new_filename = f"{prefix}_{img_counter:04d}_{image_name}.jpg"
        save_path = os.path.join(output_dir, new_filename)
        txt_path = os.path.join(txt_path_prefix, new_filename)

        cv2.imwrite(save_path, img)
        txt_paths.append(txt_path)
        touched_paths.append(path)
        img_counter += 1

        if verbose:
            if idx % 100 == 0:
                print(f"----------------------------------------------------------------------------------")
                print(f"[Path left: {len(path_list) - idx}][idx: {idx}][path_list: {len(path_list)}] ")
                print(f"----------------------------------------------------------------------------------")

        if verbose:
            print(f"[OK][{img_counter}] Saved: {save_path}")
        
        
    for path in touched_paths:
        path_list.remove(path)

    return txt_paths



if __name__ == '__main__':
    # Open and load the YAML file
    with open('options/dataset_query.yml', 'r') as file:
        opt = yaml.safe_load(file)

    img_paths = pull_image_paths(opt['pull_paths'], opt['deep_pull'], opt['img_types'])
    img_paths = shuffle_image_paths(img_paths)

    max_images = opt['max_images']
    if len(img_paths) < max_images and max_images > 0:
        print(f"[Warning] Not enough images found. Found {len(img_paths)}, but max_images is set to {max_images}. Using all available images.")
        max_images = len(img_paths)

    max_train_images = max_images * opt['split_ratio']
    max_val_images = max_images - max_train_images

    dataset_store_path = opt['store_path']
    dataset_name = opt['name']

    train_path = os.path.join(dataset_store_path, dataset_name, 'train')
    val_path = os.path.join(dataset_store_path, dataset_name, 'val')
    train_txt_path = os.path.join(dataset_store_path, dataset_name, 'train.txt')
    val_txt_path = os.path.join(dataset_store_path, dataset_name, 'val.txt')

    txt_path_prefix = opt['txt_path_prefix']

    image_opt = opt['image']
    filter_opt = image_opt['filter']

    train_txt_file = load_and_store_images(
            img_paths,
            train_path,
            max_train_images,
            os.path.join(txt_path_prefix, 'train/'),
            opt['print_steps'],
            #filter
            filter_opt['max_img_size'],
            filter_opt['min_img_size'],
            filter_opt['max_black_percentage'],
            filter_opt['black_threshold'],
            filter_opt['max_white_percentage'],
            filter_opt['white_threshold'],
            #cut    
            image_opt['horizontal_cut'],
            image_opt['horizontal_cut_size'],
            image_opt['horizontal_cut_direction'],
            image_opt['vertical_cut'],
            image_opt['vertical_cut_size'],
            image_opt['vertical_cut_direction'],
            #squezze
            image_opt['resize'],
            image_opt['resize_until'],
            prefix='train_',
            verbose=True
            )
    
    os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
    with open(train_txt_path, 'w') as f:
        for path in train_txt_file:
            f.write(f"{path}\n")

    val_txt_file = load_and_store_images(
            img_paths,
            val_path,
            max_val_images,
            os.path.join(txt_path_prefix, 'val/'),
            opt['print_steps'],
            #filter
            filter_opt['max_img_size'],
            filter_opt['min_img_size'],
            filter_opt['max_black_percentage'],
            filter_opt['black_threshold'],
            filter_opt['max_white_percentage'],
            filter_opt['white_threshold'],
            #cut    
            image_opt['horizontal_cut'],
            image_opt['horizontal_cut_size'],
            image_opt['horizontal_cut_direction'],
            image_opt['vertical_cut'],
            image_opt['vertical_cut_size'],
            image_opt['vertical_cut_direction'],
            #squezze
            image_opt['resize'],
            image_opt['resize_until'],
            prefix='val_',
            verbose=True
        )
    
    
    os.makedirs(os.path.dirname(val_txt_path), exist_ok=True)

    with open(val_txt_path, 'w') as f:
        for path in val_txt_file:
            f.write(f"{path}\n")
