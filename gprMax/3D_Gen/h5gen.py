import h5py
import math
from PIL import Image
import numpy as np
import os
import argparse

type = "defect"  # defect or healthy
iteration = 0
# Constants

output_directory = 'png'
L = 1
H = 0.3
count = 51
res = 200
input_dir = 'png'
output_dir = 'h5'
color_map = {
    (0, 0, 0, 0): -1,
    (255, 255, 0, 255): 0,  # Yellow
    (255, 51, 0, 255): 1  # Red
}

parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--mode', type=int,
                    help='png for generate rotating images, h5 for generate h5 from png')
parser.add_argument('--start', type=int, help='Starting index value')
parser.add_argument('--count', type=int, help='Iteration count value')
parser.add_argument('--prefix', type=str, help='File prefix name')
args = parser.parse_args()

mode = args.mode
starting_index = args.start
iteration_cnt = args.count
prefix = args.prefix


def find_angle(current_position, L, H):
    if current_position < L:
        L3 = L - current_position
        current_angle = math.atan(H / L3) * 180 / math.pi
    elif current_position > L:
        L3 = current_position - L
        current_angle = (math.atan(H / L3)) * 180 / math.pi
        current_angle = 180 - current_angle
    else:  # exactly middle
        current_angle = 90

    return 90 - current_angle

# Generates rotating png images


def generate_rotated_images(input_image_path, output_directory, L, H, count, type, iteration):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    mid = L / 2
    step = (L / (count - 1))

    # Open the original image
    original_image = Image.open(input_image_path)
    dist = []
    for i in range(count):
        dist.append(math.sqrt(H**2 + (mid - i * step)**2))
        angle = find_angle(i * step, mid, H)

        # Rotate the original image by the calculated angle
        rotated_image = original_image.rotate(angle)

        # Save the rotated image with the angle in the filename
        output_image_path = os.path.join(
            output_directory, f'{type}{iteration}_{i}.png')
        rotated_image.save(output_image_path)
    dist = np.array(dist)
    rd_dist = np.round(dist / 0.002) * 0.002
    rd_dist = [(round(x, len(str(0.002).split('.')[1]))) for x in rd_dist]
    return rd_dist


def is_color_approx(pixel_color, target_color, tolerance=10):
    return all(abs(pixel_color[i] - target_color[i]) <= tolerance for i in range(len(pixel_color)))


if mode == 'png':
    for iteration in range(starting_index, starting_index + iteration_cnt):
        input_image_path = f'image/healthy/healthy{iteration}.png'
        pos = generate_rotated_images(
            input_image_path, output_directory, L, H, count, "healthy", iteration)
        input_image_path = f'image/defect/defect{iteration}.png'
        pos = generate_rotated_images(
            input_image_path, output_directory, L, H, count, "defect", iteration)
        np.array(pos)

elif mode == 'h5':
    # Creates h5 from png files:
    for iteration in range(starting_index, starting_index + iteration_cnt):
        for image_name in os.listdir(input_dir):
            if image_name.endswith(".png"):
                # Load the PNG image
                image_path = os.path.join(input_dir, image_name)
                img = Image.open(image_path)

                # Resize the image
                img_resized = img.resize((res, res))

                # Initialize the array with -1 for all values
                arr_2d = np.full((res, res), -1, dtype=int)

                # Iterate through each pixel and assign values based on the color map
                for y in range(res):
                    for x in range(res):
                        pixel_color = img_resized.getpixel((x, y))

                        # Check if the pixel color is approximately equal to a color in the color map
                        matching_color = None
                        for target_color, value in color_map.items():
                            if is_color_approx(pixel_color[:-1], target_color[:-1], tolerance=102):
                                matching_color = value
                                break

                        if matching_color is not None:
                            arr_2d[y, x] = matching_color

                arr_3d = np.expand_dims(arr_2d, axis=2)

                # Extract the base filename without the extension
                base_filename = os.path.splitext(
                    os.path.basename(image_path))[0]

                # Create the new filename with the resolution
                filename = f"{base_filename}_{res}.h5"

                filepath = os.path.join(output_dir, filename)

                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Create a dataset within the 'data' group and store the array
                with h5py.File(filepath, 'w') as file:
                    dset = file.create_dataset("data", data=arr_3d)

                    # Add a root attribute with the name 'dx_dy_dz'
                    file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)

else:
    print("Please specify the mode, either h5 or png.")
