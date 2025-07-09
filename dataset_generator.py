import yaml
import glob
import cv2
import os
import math
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

def resize_image(img, resize=True, resize_horizontal=512, resize_vertical=512):
    """
    Resizes the image to the specified width and height if resizing is enabled.

    Args:
        img (numpy.ndarray): The input image.
        resize (bool): Whether to apply resizing.
        resize_horizontal (int): Target width.
        resize_vertical (int): Target height.

    Returns:
        numpy.ndarray: Resized image if needed, otherwise original.
    """
    if not resize:
        return img

    # Resize image to target dimensions
    img = cv2.resize(img, (resize_horizontal, resize_vertical), interpolation=cv2.INTER_AREA)
    return img
    return img

def combine_to_grid(images, grid_size):
    if len(images) == 0:
        raise ValueError("Image list is empty.")
    if grid_size != len(images):
        raise ValueError("grid_size must match number of images.")

    # Determine grid shape (rows x cols)
    grid_rows = int(math.floor(math.sqrt(grid_size)))
    grid_cols = int(math.ceil(grid_size / grid_rows))

    # Get image shape
    img_h, img_w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1

    # Pad with black images if necessary
    num_missing = grid_rows * grid_cols - grid_size
    if num_missing > 0:
        pad_img = np.zeros_like(images[0])
        images += [pad_img] * num_missing

    # Combine images
    rows = []
    for i in range(grid_rows):
        row_imgs = images[i * grid_cols:(i + 1) * grid_cols]
        row = np.hstack(row_imgs)
        rows.append(row)

    grid_image = np.vstack(rows)
    return grid_image


def load_and_store_images(
        path_list,
        output_dir,
        max_images,
        txt_path_prefix,
        print_steps,

        # Filter
        max_img_size,
        min_img_size,
        max_black_percentage,
        black_threshold,
        max_white_percentage,
        white_threshold,

        # Cut
        horizontal_cut,
        horizontal_cut_size,
        horizontal_cut_direction,
        vertical_cut,
        vertical_cut_size,
        vertical_cut_direction,

        # Resize
        resize,
        resize_horizontal=512,
        resize_vertical=512,

        # Grid
        grid_square=False,
        grid_square_size=4,
        grid_resize_horizontal=512,
        grid_resize_vertical=512,

        prefix='img_',
        verbose=True):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    txt_paths = []
    touched_paths = []
    group_buffer = []
    grid_counter = 0

    for idx, path in enumerate(path_list):
        if grid_counter >= max_images:
            break

        img = cv2.imread(path)
        if img is None:
            if verbose:
                print(f"[Warning] Failed to load: {path}")
            continue

        # Size filter
        if filter_img_size(img, min_img_size, max_img_size):
            if verbose:
                print(f"[Skip] Image size not in range: {path}")
            continue

        # Crop and resize
        img = cut_image_to_size(
            img,
            horizontal_cut=horizontal_cut,
            horizontal_cut_size=horizontal_cut_size,
            horizontal_cut_direction=horizontal_cut_direction,
            vertical_cut=vertical_cut,
            vertical_cut_size=vertical_cut_size,
            vertical_cut_direction=vertical_cut_direction
        )

        img = resize_image(
            img,
            resize=resize,
            resize_horizontal=resize_horizontal,
            resize_vertical=resize_vertical
        )

        # Check if image has valid size
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            if verbose:
                print(f"[Skip] Image became invalid after cropping: {path}")
            continue

        # Brightness filter
        brightness_check = filter_img_brightness(
            img,
            max_black_percentage,
            black_threshold,
            max_white_percentage,
            white_threshold
        )

        if brightness_check != 0:
            if verbose:
                reason = "too dark" if brightness_check == -1 else "too bright"
                print(f"[Skip] Image {reason}: {path}")
            continue

        # Add to buffer
        group_buffer.append(img)
        touched_paths.append(path)

        if verbose:
            print(f"[ADD] Image added to buffer: {path} (Buffer size: {len(group_buffer)}/{grid_square_size})")

        # Build and save grid when full
        if len(group_buffer) == grid_square_size:
            # Filter out invalid images before combining
            valid_images = [img for img in group_buffer if img.shape[0] > 0 and img.shape[1] > 0]
            
            if len(valid_images) != grid_square_size:
                if verbose:
                    print(f"[Skip] Invalid image(s) in group. Expected {grid_square_size}, got {len(valid_images)} valid.")
                group_buffer = []  # Clear and move on
                continue
            grid_img = combine_to_grid(group_buffer, grid_square_size)
            grid_img = resize_image(grid_img, True, grid_resize_horizontal, grid_resize_vertical)

            brightness_check = filter_img_brightness(
                grid_img,
                max_black_percentage,
                black_threshold,
                max_white_percentage,
                white_threshold
            )

            if brightness_check == 0:
                new_filename = f"{prefix}_group_{grid_counter:04d}.jpg"
                save_path = os.path.join(output_dir, new_filename)
                txt_path = os.path.join(txt_path_prefix, new_filename)

                cv2.imwrite(save_path, grid_img)
                grid_counter += 1
                txt_paths.append(txt_path)

                if verbose:
                    print(f"[OK] Saved grid image: {save_path}")
            else:
                if verbose:
                    reason = "GRID too dark" if brightness_check == -1 else "GRID too bright"
                    print(f"[Skip] {reason}: group {grid_counter}")
                continue

            group_buffer = []

        if verbose and idx % print_steps == 0:
            print(f"[Progress] Processed {idx+1}/{len(path_list)}")

    # Handle leftover images
    if grid_square and len(group_buffer) > 0:
        if verbose:
            print(f"[FINAL] Saving incomplete group of {len(group_buffer)} images")

        grid_img = combine_to_grid(group_buffer, grid_square_size)
        grid_img = resize_image(grid_img, True, grid_resize_horizontal, grid_resize_vertical)

        brightness_check = filter_img_brightness(
            grid_img,
            max_black_percentage,
            black_threshold,
            max_white_percentage,
            white_threshold
        )

        if brightness_check == 0:
            new_filename = f"{prefix}_group_{grid_counter:04d}.jpg"
            save_path = os.path.join(output_dir, new_filename)
            txt_path = os.path.join(txt_path_prefix, new_filename)

            cv2.imwrite(save_path, grid_img)
            txt_paths.append(txt_path)

            if verbose:
                print(f"[OK] Saved final (partial) grid image: {save_path}")
        else:
            if verbose:
                reason = "FINAL grid too dark" if brightness_check == -1 else "FINAL grid too bright"
                print(f"[Skip] {reason}: final group")

    # Cleanup used paths
    for path in touched_paths:
        if path in path_list:
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
            # General options
            print_steps=opt['print_steps'],

            # Filter options
            max_img_size=filter_opt['max_img_size'],
            min_img_size=filter_opt['min_img_size'],
            max_black_percentage=filter_opt['max_black_percentage'],
            black_threshold=filter_opt['black_threshold'],
            max_white_percentage=filter_opt['max_white_percentage'],
            white_threshold=filter_opt['white_threshold'],

            # Cut options
            horizontal_cut=image_opt['horizontal_cut'],
            horizontal_cut_size=image_opt['horizontal_cut_size'],
            horizontal_cut_direction=image_opt['horizontal_cut_direction'],
            vertical_cut=image_opt['vertical_cut'],
            vertical_cut_size=image_opt['vertical_cut_size'],
            vertical_cut_direction=image_opt['vertical_cut_direction'],

            # Squeeze (resize) options
            resize=image_opt['resize'],
            resize_horizontal=image_opt['resize_horizontal'],
            resize_vertical=image_opt['resize_vertical'],

            # Grid options
            grid_square=image_opt['grid_square'],
            grid_square_size=image_opt['grid_square_size'],
            grid_resize_horizontal=image_opt['grid_resize_horizontal'],
            grid_resize_vertical=image_opt['grid_resize_vertical'],

            #prefix
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
            # General options
            print_steps=opt['print_steps'],

            # Filter options
            max_img_size=filter_opt['max_img_size'],
            min_img_size=filter_opt['min_img_size'],
            max_black_percentage=filter_opt['max_black_percentage'],
            black_threshold=filter_opt['black_threshold'],
            max_white_percentage=filter_opt['max_white_percentage'],
            white_threshold=filter_opt['white_threshold'],

            # Cut options
            horizontal_cut=image_opt['horizontal_cut'],
            horizontal_cut_size=image_opt['horizontal_cut_size'],
            horizontal_cut_direction=image_opt['horizontal_cut_direction'],
            vertical_cut=image_opt['vertical_cut'],
            vertical_cut_size=image_opt['vertical_cut_size'],
            vertical_cut_direction=image_opt['vertical_cut_direction'],

            # Squeeze (resize) options
            resize=image_opt['resize'],
            resize_horizontal=image_opt['resize_horizontal'],
            resize_vertical=image_opt['resize_vertical'],

            # Grid options
            grid_square=image_opt['grid_square'],
            grid_square_size=image_opt['grid_square_size'],
            grid_resize_horizontal=image_opt['grid_resize_horizontal'],
            grid_resize_vertical=image_opt['grid_resize_vertical'],
            #prefix
            prefix='val_',
            verbose=True
        )
    
    
    os.makedirs(os.path.dirname(val_txt_path), exist_ok=True)

    with open(val_txt_path, 'w') as f:
        for path in val_txt_file:
            f.write(f"{path}\n")
