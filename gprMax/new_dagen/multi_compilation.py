import concurrent.futures
import subprocess
import matplotlib.pyplot as plt
import shutil
import glob
import h5py
from PIL import Image
import cv2
import math
import numpy as np
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Script description')

os.chdir('/kaggle/working/FYP/gprMax/new_dagen')


# Directories Names
working_dir = 'working'
output_dir = 'output'
input_path = '/kaggle/input/multilayers-0-10'

parser.add_argument('--start', type=int, help='Starting index value')

parser.add_argument('--count', type=int, help='Iteration count value')

args = parser.parse_args()
starting_index = args.start
iteration_cnt = args.count

# Create the output directory if it doesn't exist
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, 'png'))
    os.makedirs(os.path.join(output_dir, 'geometry'))
    os.makedirs(os.path.join(output_dir, 'out'))

air_in_file = f"""#python:
src_to_trunk_center = 0.5
radius =  0.3

resolution = 0.002
time_window = 2e-8 #do a study
pml_cells = 25

x_gap = 0.05
y_gap = 0.02
src_to_pml = 0.05
src_to_rx = 0.1


# Derived Parameters
src_to_trunk = src_to_trunk_center-radius
diameter = radius* 2

pml = resolution * pml_cells
sharp_domain = [diameter, diameter + src_to_trunk]
domain = [sharp_domain[0] + pml * 2 + x_gap * 2, sharp_domain[1] +  pml * 2 + y_gap + src_to_pml]

trunk_center = [radius + pml + x_gap, src_to_trunk + radius + pml + src_to_pml]
src_position = [trunk_center[0] - (src_to_rx / 2), pml + src_to_pml, 0]
rx_position = [src_position[0] + src_to_rx, src_position[1], 0]

print('#title: Air Scan')
print("#domain: {{:.3f}} {{:.3f}} 0.002".format(domain[0], domain[1]))
print("#dx_dy_dz: {{}} {{}} {{}}".format(resolution, resolution, resolution))
print("#time_window: {{}}".format(time_window))
print()
print('#pml_cells: {{}} {{}} 0 {{}} {{}} 0'.format(pml_cells, pml_cells, pml_cells, pml_cells))
print()
print('#waveform: ricker 1 1e9 my_ricker')
print("#hertzian_dipole: z {{:.3f}} {{:.3f}} {{:.3f}} my_ricker".format(src_position[0], src_position[1], src_position[2]))
print("#rx: {{:.3f}} {{:.3f}} {{:.3f}}".format(rx_position[0], rx_position[1], rx_position[2]))
#end_python:
"""

# Specify the file path where you want to save the Python code
air_path = f"{working_dir}/air.in"

# Open the file in write mode
with open(air_path, "w") as file:
    # Write the Python code to the file
    file.write(air_in_file)

# Construct the command
command = f"python -m gprMax {air_path} -n 1 -gpu"

# Run the command
try:
    subprocess.run(command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: running command: {command} {e}")


# %%time
for idx in range(starting_index, starting_index + iteration_cnt):
    # GET THE CSV DATA FOR EACH IDX
    print(idx)
    csv_path = f"{input_path}/csv"

    # Specify the headers as a list
    headers = ["idx", "x_cavity", "y_cavity", "r_cavity", "trunk_dist",
               "radius", "eps_bark", "eps_sapwood", "eps_heartwood", "eps_decay"]
    # Function to find the folder and CSV file for a given number

    def find_csv_for_number(number, folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)

        # Iterate through the files to find the matching CSV file
        for file in files:
            if file.startswith("data_") and file.endswith(".csv"):
                try:
                    # Extract the lower and upper bounds from the filename
                    _, lower, upper = file.split("_")
                    lower = int(lower)
                    upper = int(upper.split(".")[0])  # Remove ".csv" extension

                    # Check if the given number is within the range
                    if lower <= number <= upper:
                        # Build the full path to the CSV file
                        csv_file_path = os.path.join(folder_path, file)
                        return csv_file_path
                except ValueError:
                    continue  # Skip files with invalid naming

        # Return None if no matching CSV file is found
        return None
    # Usage
    csv_file_path = find_csv_for_number(idx, csv_path)

    if csv_file_path:
        # Now you can read the CSV file using your preferred method (e.g., pandas)
        df = pd.read_csv(csv_file_path, header=None, names=headers)
    else:
        print(f"No CSV file found for number {idx}")

    # h5 gen
    h5_output_dir = os.path.join(os.getcwd(), 'h5')

    if not os.path.exists(h5_output_dir):
        os.makedirs(h5_output_dir)

    # Function to find the angle based on the currentPosition and L
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

    def generate_rotated_images(input_image_path, L, H, count):

        mid = L / 2
        step = (L / (count - 1))

        # Load the image with alpha channel
        original_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        dist = []
        rotated_images = []

        for i in range(count):
            dist.append(math.sqrt(H**2 + (mid - i * step)**2))
            angle = find_angle(i * step, mid, H)

            center = (original_image.shape[1] //
                      2, original_image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                original_image, rotation_matrix, (original_image.shape[1], original_image.shape[0]))

            rotated_images.append(rotated_image)

        dist = np.array(dist)
        rd_dist = np.round(dist / 0.002) * 0.002
        rd_dist = [(round(x, len(str(0.002).split('.')[1]))) for x in rd_dist]

        return rd_dist, rotated_images

    color_map = {
        (0, 0, 0, 0): -1,
        (0, 223, 255, 255): 0,  # Yellow
        (0, 159, 255, 255): 1,  # Yellow
        (0, 95, 255, 255): 2,  # Yellow
        (0, 31, 255, 255): 3  # Red
    }

    # Function to check if a pixel color is approximately equal to a given color
    def is_color_approx(pixel_color, target_color, tolerance=10):
        return np.all(np.abs(pixel_color - target_color) <= tolerance, axis=-1)

    # Function to process a single image and save it as an HDF5 file
    def process_image(img, output, h5_output_dir, res):
        # Resize the image
        img_resized = cv2.resize(img, (res, res))

        # Convert the resized image to a NumPy array
        img_array = np.array(img_resized)

        # Initialize the array with -1 for all values
        arr_2d = np.full((res, res), -1, dtype=int)

        for target_color, value in color_map.items():
            # Create a mask of pixels that match the target_color
            mask = is_color_approx(
                img_array, np.array(target_color), tolerance=32)

            # Assign the corresponding value to matching pixels
            arr_2d[mask] = value

        arr_3d = np.expand_dims(arr_2d, axis=2)

        filepath = os.path.join(h5_output_dir, output)

        # Create the output directory if it doesn't exist
        if not os.path.exists(h5_output_dir):
            os.makedirs(h5_output_dir)

        # Create a dataset within the 'data' group and store the array
        with h5py.File(filepath, 'w') as file:
            dset = file.create_dataset("data", data=arr_3d)

            # Add a root attribute with the name 'dx_dy_dz'
            file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)

    # GENERATE H5 FILES
    L = 1
    count = 51
    H = df.loc[df['idx'] == idx, 'trunk_dist'].values[0]
    res = int(
        np.round((df.loc[df['idx'] == idx, 'radius'].values[0]/0.002 * 2)))

    for mode in ['healthy', 'defect']:
        target = f'{mode}{idx}'
        input_image_path = f'{input_path}/image/{mode}/{target}.png'
        print(input_image_path)

        # Generate rotated array of images
        pos, pngs = generate_rotated_images(input_image_path, L, H, count)
        # List all files in the input directory
        image_files = [image for image in pngs]
        output_names = [f'{mode}{idx}_{i}.h5' for i in range(len(image_files))]

        # Create a ThreadPoolExecutor to process images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_image, image_files, output_names, [
                         h5_output_dir] * len(image_files), [res] * len(image_files))

    # RUN GPRMAX FOR 3 TYPES
    for runtype in ['healthy', 'cavity', 'decay']:
        print(runtype)
        if runtype == 'healthy':
            mode = 'healthy'
        elif runtype == 'cavity' or runtype == 'decay':
            mode = 'defect'

        in_file = f"""#python:
b_scan_cnt = {count} 
pos = {pos}
radius =  {df.loc[df['idx'] == idx, 'radius'].values[0]}

src_to_trunk_center = pos[current_model_run - 1]
resolution = 0.002
time_window = 2e-8 #do a study
pml_cells = 25

x_gap = 0.05
y_gap = 0.02
src_to_pml = 0.05
src_to_rx = 0.1


# Derived Parameters
src_to_trunk = src_to_trunk_center-radius
diameter = radius* 2

pml = resolution * pml_cells
sharp_domain = [diameter, diameter + src_to_trunk]
domain = [sharp_domain[0] + pml * 2 + x_gap * 2, sharp_domain[1] +  pml * 2 + y_gap + src_to_pml]

trunk_center = [radius + pml + x_gap, src_to_trunk + radius + pml + src_to_pml]
src_position = [trunk_center[0] - (src_to_rx / 2), pml + src_to_pml, 0]
rx_position = [src_position[0] + src_to_rx, src_position[1], 0]

print('#title: {runtype.upper()} {idx} Rotating Straight Scan')
print("#domain: {{:.3f}} {{:.3f}} 0.002".format(domain[0], domain[1]))
print("#dx_dy_dz: {{}} {{}} {{}}".format(resolution, resolution, resolution))
print("#time_window: {{}}".format(time_window))
print()
print('#pml_cells: {{}} {{}} 0 {{}} {{}} 0'.format(pml_cells, pml_cells, pml_cells, pml_cells))
print()
print("#geometry_objects_read: {{:.3f}} {{:.3f}} 0 h5/{mode}{idx}_{{}}.h5 materials.txt".format((trunk_center[0]) - radius, (trunk_center[1]) - radius, current_model_run - 1)) 
print()
print('#waveform: ricker 1 1e9 my_ricker')
print("#hertzian_dipole: z {{:.3f}} {{:.3f}} {{:.3f}} my_ricker".format(src_position[0], src_position[1], src_position[2]))
print("#rx: {{:.3f}} {{:.3f}} {{:.3f}}".format(rx_position[0], rx_position[1], rx_position[2]))
if (current_model_run == (b_scan_cnt - 1)/2):
    print("#geometry_objects_write: {{:.3f}} {{:.3f}} 0 {{:.3f}} {{:.3f}} 0.002 {runtype}".format(trunk_center[0] - radius, trunk_center[1] - src_to_trunk_center, trunk_center[0] + radius, trunk_center[1] + radius))
#end_python:
"""
        defect_eps = 1
        if(runtype == 'decay'):
            defect_eps = df.loc[df['idx'] == idx, 'eps_decay'].values[0]

        material_file = f"""#material: {df.loc[df['idx'] == idx, 'eps_bark'].values[0]} 0 1 0 bark
#material: {df.loc[df['idx'] == idx, 'eps_sapwood'].values[0]} 0 1 0 sapwood
#material: {df.loc[df['idx'] == idx, 'eps_heartwood'].values[0]} 0 1 0 heartwood
#material: {defect_eps} 0 1 0 defect
        """
        # Specify the file path where you want to save the Python code
        in_path = f"{working_dir}/{runtype}.in"

        # Open the file in write mode
        with open(in_path, "w") as file:
            # Write the Python code to the file
            file.write(in_file)

        # Specify the file path where you want to save the Python code
        mat_path = f"{working_dir}/materials.txt"

        # Open the file in write mode
        with open(mat_path, "w") as file:
            # Write the Python code to the file
            file.write(material_file)

        # Construct the command
        command = f"python -m gprMax {in_path} -n {count} -gpu"

        # Run the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: running command: {command} {e}")

        # Construct the command
        command = f"python -m tools.outputfiles_merge {working_dir}/{runtype} --remove-files"

        # Run the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: running command: {command} {e}")

        # Process Outputs
        with h5py.File(f'{working_dir}/air.out', 'r') as f:
            Ez0 = f['rxs']['rx1']['Ez'][()]

        Ez0 = Ez0[:, np.newaxis]  # Add a new axis

        Ez0 = np.repeat(Ez0, count, axis=1)
        with h5py.File(f'{working_dir}/{runtype}.h5', 'r') as f2:
            dset = f2['data'][()]

        # Set the desired width and height of the figure
        fig_width = dset.shape[1] * 0.01
        fig_height = dset.shape[0] * 0.01

        # Create the figure and axes objects with the desired size and arrangement
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot dset on the left subplot
        # For 3D
        # axs[0].imshow(np.transpose(dset[310,:,:], axes=(1, 0)), cmap='viridis')
        ax.imshow(dset, cmap='gray')
        ax.invert_yaxis()

        # Display the image
        plt.axis('off')
        plt.savefig(f"{output_dir}/geometry/{runtype}{idx}.png",
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load the saved image
        original_image = Image.open(
            f"{output_dir}/geometry/{runtype}{idx}.png")

        # Set the size of the larger canvas
        canvas_width = 1500  # Specify your desired width
        canvas_height = 1500  # Specify your desired height

        # Create a blank white canvas
        canvas = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))

        y_offset = (canvas_height - original_image.height) // 2

        # Paste the original image onto the top-left corner of the canvas
        canvas.paste(original_image, (0, y_offset))

        # Save or display the canvas with the image
        canvas.save(f"{output_dir}/geometry/{runtype}{idx}.png", format='png')

        with h5py.File(f'{working_dir}/{runtype}_merged.out', 'r') as f:
            bscan = np.subtract(f['rxs']['rx1']['Ez'][()], Ez0)

        # Set the desired width and height of the figure
        fig_width = 5
        fig_height = 5

        # Create the figure and axes objects with the desired size and arrangement
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot dset on the left subplot
        # For 3D
        # axs[0].imshow(np.transpose(dset[310,:,:], axes=(1, 0)), cmap='viridis')
        ax.imshow(bscan, cmap='gray', aspect='auto')

        # Display the image
        plt.axis('off')
        plt.savefig(f"{output_dir}/png/{runtype}{idx}_bscan.png",
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        # Copy the source file to the destination
        shutil.copy(f'{working_dir}/{runtype}_merged.out',
                    f'{output_dir}/out/{runtype}{idx}.out')
    if idx % 10 == 0:
        # Delete h5 rotated files
        files = glob.glob(os.path.join(h5_output_dir, '*.h5'))
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")
