# # %% [markdown]
# # # Part 2: Generate required files for each idx, mode

# # %% [code] {"jupyter":{"outputs_hidden":false}}
# import os
# !git clone https://github.com/nathandangg/FYP.git
# os.chdir('/kaggle/working/FYP/gprMax')
# !conda env update -f conda_env.yml
# !conda activate gprMax
# !python setup.py build
# !python setup.py install
# os.chdir('/kaggle/working/FYP/gprMax/new_dagen')

# # %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import pandas as pd
import numpy as np
import math
from PIL import Image
import h5py
import glob
import shutil
import matplotlib.pyplot as plt
import subprocess

# Directories Names
working_dir = 'working'
output_dir = 'output'
input_path = '/kaggle/input/multilayers-0-10'

for idx in range(10):
    for type in ['healthy', 'cavity', 'decay']:
        if type == 'healthy':
            mode = 'healthy'
        elif type == 'cavity':
            mode = 'defect'
        elif type == 'decay':
            mode = 'defect'
        target = f'{mode}{idx}'
        csv_path = f"{input_path}/csv"
        input_image_path = f'{input_path}/image/{mode}/{target}.png'

        # Specify the headers as a list
        headers = ["idx", "CenBcavity_X", "CenBcavity_Y", "r_cavity", "trunk_dist", "radius", "eps_bark", "eps_sapwood", "eps_heartwood", "eps_decay"]
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

        csv_file_path = find_csv_for_number(idx, csv_path)

        if csv_file_path:
            # Now you can read the CSV file using your preferred method (e.g., pandas)
            df = pd.read_csv(csv_file_path, header=None, names=headers)
        else:
            print(f"No CSV file found for number {idx}")

        df.head()
        # Create the output directory if it doesn't exist
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir, 'png'))
            os.makedirs(os.path.join(output_dir, 'geometry'))
            os.makedirs(os.path.join(output_dir, 'out'))

        H = df.loc[df['idx'] == idx, 'trunk_dist'].values[0] # TODO: read from csv
        res = int(np.round((df.loc[df['idx'] == idx, 'radius'].values[0]/0.002 * 2))) #TODO: read from csv

        # png gen
        png_output_dir = 'png'

        L = 1
        count = 51

        # h5 gen
        print(f"Trunk area: {res} x {res} cells")
        h5_input_dir = 'png'
        h5_output_dir = 'h5'
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

        # Function to generate rotated images
        def generate_rotated_images(input_image_path, output_directory, target, L, H, count):
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            mid = L / 2
            step = (L / (count - 1))

            # Open the original image
            original_image = Image.open(input_image_path)
            dist = []
            rotated_images = []
            for i in range(count):
                dist.append(math.sqrt(H**2 + (mid - i * step)**2))
                angle = find_angle(i * step, mid, H)

                # Rotate the original image by the calculated angle
                rotated_image = original_image.rotate(angle)
                rotated_images.append(rotated_image)

            # Save the rotated image with the angle in the filename
            # output_image_path = os.path.join(output_directory, f'{target}_{i}.png')
            # rotated_image.save(output_image_path)
            for i, rotated_image in enumerate(rotated_images):
                output_image_path = os.path.join(output_directory, f'{target}_{i}.png')
                rotated_image.save(output_image_path)

            dist = np.array(dist)
            rd_dist = np.round(dist / 0.002) * 0.002
            rd_dist = [(round(x, len(str(0.002).split('.')[1]))) for x in rd_dist]
            return rd_dist

        pos = generate_rotated_images(input_image_path, png_output_dir, target, L, H, count)
        np.array(pos)
        # Color map
        color_map = {
            (0, 0, 0, 0): -1,
            (255, 223, 0, 255): 0,  # Yellow
            (255, 159, 0, 255): 1,  # Yellow
            (255, 95, 0, 255): 2,  # Yellow
            (255, 31, 0, 255): 3  # Red
        }

        # Function to check if a pixel color is approximately equal to a given color
        def is_color_approx(pixel_color, target_color, tolerance=10):
            return np.all(np.abs(pixel_color - target_color) <= tolerance, axis=-1)

        # Iterate through all files in the input directory
        for image_name in os.listdir(h5_input_dir):
            if image_name.endswith(".png"):
                # Load the PNG image
                image_path = os.path.join(h5_input_dir, image_name)
                img = Image.open(image_path)

                # Resize the image
                img_resized = img.resize((res, res))

                # Convert the resized image to a NumPy array
                img_array = np.array(img_resized)

                arr_2d = np.full((res, res), -1, dtype=int)  # Initialize the array with -1 for all values

                for target_color, value in color_map.items():
                    # Create a mask of pixels that match the target_color
                    mask = is_color_approx(img_array, np.array(target_color), tolerance=32)

                    # Assign the corresponding value to matching pixels
                    arr_2d[mask] = value

                arr_3d = np.expand_dims(arr_2d, axis=2)

                # Extract the base filename without the extension
                base_filename = os.path.splitext(os.path.basename(image_path))[0]

                # Create the new filename with the resolution
                filename = f"{base_filename}.h5"

                filepath = os.path.join(h5_output_dir, filename)

                # Create the output directory if it doesn't exist
                if not os.path.exists(h5_output_dir):
                    os.makedirs(h5_output_dir)

                # Create a dataset within the 'data' group and store the array
                with h5py.File(filepath, 'w') as file:
                    dset = file.create_dataset("data", data=arr_3d)

                    # Add a root attribute with the name 'dx_dy_dz'
                    file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)

        air_in_file = f"""#python:
        src_to_trunk_center = {pos[int((count-1)/2)]}
        radius =  {df.loc[df['idx'] == idx, 'radius'].values[0]}

        resolution = 0.002
        time_window = 3e-8 #do a study
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
        in_file = f"""#python:
        b_scan_cnt = {count} 
        pos = {pos}
        radius =  {df.loc[df['idx'] == idx, 'radius'].values[0]}

        src_to_trunk_center = pos[current_model_run - 1]
        resolution = 0.002
        time_window = 3e-8 #do a study
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

        print('#title: Rotating Straight Scan')
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
            print("#geometry_objects_write: {{:.3f}} {{:.3f}} 0 {{:.3f}} {{:.3f}} 0.002 {mode}".format(trunk_center[0] - radius, trunk_center[1] - src_to_trunk_center, trunk_center[0] + radius, trunk_center[1] + radius))
        #end_python:
        """
        defect_eps = 1
        if(type == 'decay'):
            defect_eps = df.loc[df['idx'] == idx, 'eps_decay'].values[0]
        material_file = f"""
        #material: {df.loc[df['idx'] == idx, 'eps_bark'].values[0]} 0 1 0 bark
        #material: {df.loc[df['idx'] == idx, 'eps_sapwood'].values[0]} 0 1 0 sapwood
        #material: {df.loc[df['idx'] == idx, 'eps_heartwood'].values[0]} 0 1 0 heartwood
        #material: {defect_eps} 0 1 0 cavity
        """
        # Specify the file path where you want to save the Python code
        in_path = f"{working_dir}/{mode}.in"

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

        # Specify the file path where you want to save the Python code
        air_path = f"{working_dir}/air.in"

        # Open the file in write mode
        with open(air_path, "w") as file:
            # Write the Python code to the file
            file.write(air_in_file)

        os.chdir('/kaggle/working/FYP/gprMax/new_dagen')
        # api(air_path, n= 1, geometry_only=False, geometry_fixed=False)
        # !python -m gprMax working/air.in -n 1

        # Construct the command
        command = f"python -m gprMax {air_path} -n 1"

        # Run the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: running command: {command} {e}")

        # api(in_path, n= count, geometry_only=False, geometry_fixed=False)
        # !python -m gprMax working/defect.in -n 1

        # Construct the command
        command = f"python -m gprMax {in_path} -n {count}"

        # Run the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: running command: {command} {e}")

        # python -m tools.outputfiles_merge user_models/cylinder_Bscan_2D
        # merge_files(f'{working_dir}/{mode}',True)
        # Construct the command
        command = f"python -m tools.outputfiles_merge {working_dir}/{mode} --remove-files"

        # Run the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: running command: {command} {e}")

        # Delete png rotated files
        files = glob.glob(os.path.join(png_output_dir, '*.png'))
        # Iterate over the list of .txt files and delete them
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        # Delete h5 rotated files
        files = glob.glob(os.path.join(h5_output_dir, '*.h5'))
        # Iterate over the list of .txt files and delete them
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        with h5py.File(f'{working_dir}/air.out', 'r') as f:
            Ez0 =  f['rxs']['rx1']['Ez'][()]

        Ez0 = Ez0[:, np.newaxis]  # Add a new axis

        Ez0 = np.repeat(Ez0, count, axis=1)
        with h5py.File(f'{working_dir}/{mode}.h5', 'r') as f2:
            dset = f2['data'][()]

        # Set the desired width and height of the figure
        fig_width = dset.shape[1] * 0.01
        fig_height = dset.shape[0] * 0.01

        # Create the figure and axes objects with the desired size and arrangement
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot dset on the left subplot
        ## For 3D
        # axs[0].imshow(np.transpose(dset[310,:,:], axes=(1, 0)), cmap='viridis')
        ax.imshow(dset, cmap='viridis')
        ax.invert_yaxis()

        # Display the image
        plt.axis('off')
        plt.savefig(f"{output_dir}/geometry/{mode}{idx}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        with h5py.File(f'{working_dir}/{mode}_merged.out', 'r') as f:
            bscan =  np.subtract(f['rxs']['rx1']['Ez'][()], Ez0)

        # Set the desired width and height of the figure
        fig_width = 2
        fig_height = 5

        # Create the figure and axes objects with the desired size and arrangement
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot dset on the left subplot
        ## For 3D
        # axs[0].imshow(np.transpose(dset[310,:,:], axes=(1, 0)), cmap='viridis')
        ax.imshow(bscan, cmap='gray', aspect = 'auto')

        # Display the image
        plt.axis('off')
        plt.savefig(f"{output_dir}/png/{type}{idx}_bscan.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        # Copy the source file to the destination
        shutil.copy(f'{working_dir}/{mode}_merged.out', f'{output_dir}/out/{type}{idx}.out')