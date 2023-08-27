from tools.plot_Bscan import mpl_plot
import csv
import random
from math import sqrt
from tools.outputfiles_merge import merge_files
from gprMax.gprMax import api
import matplotlib.colors as colors
import argparse
import h5py
import numpy as np
import os
from PIL import Image

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script description')

# Add an argument for starting_index
parser.add_argument('--start', type=int, help='Starting index value')

# Add an argument for iteration_cnt
parser.add_argument('--count', type=int, help='Iteration count value')

# Add an argument for file prefix
parser.add_argument('--prefix', type=str, help='File prefix name')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of starting_index and iteration_cnt
starting_index = args.start
iteration_cnt = args.count
prefix = args.prefix

current_folder = os.getcwd() + "/output"
current_path = os.getcwd()
#########################################


def read_csv_values(filename):
    data = []
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            row_values = []
            for cell in row:
                if cell != '':
                    row_values.append(float(cell))
            data.append(row_values)
    return data


csv_data = read_csv_values('data-config0.csv')
# for row_values in csv_data:
#     print(row_values)

file_healthy_1 = prefix + 'straight_healthy'
file_cavity_1 = prefix + 'straight_cavity'

generated_healthy_base = current_path + '/image/healthy/healthy'
generated_cavity_base = current_path + '/image/defect/defect'


def preprocess(image_path, res):
    # image_path = "TreeGen/image/defect/defect0.png"
    res = int(res)
    img = Image.open(image_path).convert('RGB')

    # Resize the image to 200x200
    img_resized = img.resize((res, res))

    # Convert the resized image to a 2D array of integers
    color_map = {
        (255, 255, 255): -1,  # White
        (255, 255, 0): 0,  # Yellow
        (255, 51, 0): 1  # Red
    }

    arr_2d = np.zeros((res, res), dtype=int)
    for y in range(res):
        for x in range(res):
            pixel_color = img_resized.getpixel((x, y))
            arr_2d[y, x] = color_map.get(pixel_color, 0)

    # Expand dimensions
    arr_3d = np.expand_dims(arr_2d, axis=2)

    base_filename = ""
    if "healthy" in image_path:
        base_filename = current_path + '/' + prefix + 'healthy.h5'
    else:
        base_filename = current_path + '/' + prefix + 'cavity.h5'

    filename = base_filename
    with h5py.File(filename, 'w') as file:
        dset = file.create_dataset("data", data=arr_3d)
        file.attrs['dx_dy_dz'] = (0.002, 0.002, 0.002)


for iteration in range(starting_index, starting_index + iteration_cnt):
    # Healthy
    pos = [0.584, 0.566, 0.55, 0.532, 0.516, 0.5, 0.484, 0.468, 0.454,
           0.438, 0.424, 0.41, 0.396, 0.384, 0.372, 0.36, 0.35, 0.34,
           0.332, 0.324, 0.316, 0.31, 0.306, 0.302, 0.3, 0.3, 0.3,
           0.302, 0.306, 0.31, 0.316, 0.324, 0.332, 0.34, 0.35, 0.36,
           0.372, 0.384, 0.396, 0.41, 0.424, 0.438, 0.454, 0.468, 0.484,
           0.5, 0.516, 0.532, 0.55, 0.566, 0.584]
    # Params
    b_scan_cnt = 51
    resolution = 0.002
    time_window = 3e-8
    pml_cells = 25

    x_gap = 0.05
    y_gap = 0.02
    src_to_pml = 0.05

    radius = csv_data[iteration][4]  # read from csv, else 0.2
    eps_trunk = csv_data[iteration][5]  # read from csv, else 10
    # can run later without init?
    src_to_trunk_center = pos[current_model_run - 1]
    src_to_rx = 0.1

    # Derived Parameters
    src_to_trunk = src_to_trunk_center-radius

    diameter = radius * 2
    pml = resolution * pml_cells
    sharp_domain = [diameter, diameter + src_to_trunk]
    domain = [sharp_domain[0] + pml * 2 + x_gap * 2,
              sharp_domain[1] + pml * 2 + y_gap + src_to_pml]

    trunk_center = [radius + pml + x_gap,
                    src_to_trunk + radius + pml + src_to_pml]
    src_position = [trunk_center[0] - (src_to_rx / 2), pml + src_to_pml, 0]
    rx_position = [src_position[0] + src_to_rx, src_position[1], 0]

    with open('{}materials.txt'.format(prefix), "w") as file:
        file.write('#material: {} 0 1 0 trunk\n'.format(eps_trunk))
        file.write('#material: 1 0 1 0 cavity\n')

    with open(file_healthy_1 + '.in', "w") as file:
        file.write('#title: Rotating Straight Scan\n')
        file.write("#domain: {:.3f} {:.3f} 0.002\n".format(
            domain[0], domain[1]))
        file.write("#dx_dy_dz: {} {} {}\n".format(
            resolution, resolution, resolution))
        file.write("#time_window: {}\n".format(time_window))
        file.write()
        file.write('#pml_cells: {} {} 0 {} {} 0\n'.format(
            pml_cells, pml_cells, pml_cells, pml_cells))
        file.write()
        file.write("#geometry_objects_read: {:.3f} {:.3f} 0 h5/healthy{}_{}_200.h5 materials.txt\n".format(
            (trunk_center[0]) - radius, (trunk_center[1]) - radius, iteration, current_model_run - 1))
        file.write()
        file.write('#waveform: ricker 1 1e9 my_ricker\n')
        file.write("#hertzian_dipole: z {:.3f} {:.3f} {:.3f} my_ricker\n".format(
            src_position[0], src_position[1], src_position[2]))
        file.write("#rx: {:.3f} {:.3f} {:.3f}".format(
            rx_position[0], rx_position[1], rx_position[2]))
        file.write("#python:")
        file.write("if (current_model_run == (b_scan_cnt - 1)/2):")
        file.write(
            "    print('#geometry_objects_write: 0 0 0 {:.3f} {:.3f} 0.002 new'.format(domain[0], domain[1])')")
        file.write("end_python:")

        api(file_healthy_1 + '.in',
            n=b_scan_cnt,
            geometry_only=False,
            geometry_fixed=False,
            # gpu=[0]
            )
    print('Done B-scan')
    merge_files(file_healthy_1, True)

    # Remove decoupling
    with open('src_only.in', "w") as file:
        file.write('#title: Rotating Straight Scan\n')
        file.write("#domain: {:.3f} {:.3f} 0.002\n".format(
            domain[0], domain[1]))
        file.write("#dx_dy_dz: {} {} {}\n".format(
            resolution, resolution, resolution))
        file.write("#time_window: {}\n".format(time_window))
        file.write('#pml_cells: {} {} 0 {} {} 0\n'.format(
            pml_cells, pml_cells, pml_cells, pml_cells))
        file.write()
        file.write('#waveform: ricker 1 1e9 my_ricker\n')
        file.write("#hertzian_dipole: z {:.3f} {:.3f} {:.3f} my_ricker\n".format(
            src_position[0], src_position[1], src_position[2]))
        file.write("#rx: {:.3f} {:.3f} {:.3f}".format(
            rx_position[0], rx_position[1], rx_position[2]))

    api('src_only.in',
        n=1,
        geometry_only=False,
        geometry_fixed=False,
        # gpu=[0]
        )
    # Merge 2 data parts
    import h5py
    import numpy as np

    filename1 = file_healthy_1 + '_merged.out'
    output_filename = prefix + 'bscan_healthy.out'
    dt = 0

    with h5py.File(filename1, 'r') as f1:
        data1 = f1['rxs']['rx1']['Ez'][()]
        dt = f1.attrs['dt']

    with h5py.File('src_only.out', 'r') as f1:
        data_source = f1['rxs']['rx1']['Ez'][()]

    src = data_source
    src = src[:, np.newaxis]
    Ez0 = np.repeat(src, b_scan_cnt, axis=1)

    merged_data_healthy = np.subtract(data1, Ez0)

    # Create a new output file and write the merged data
    with h5py.File(output_filename, 'w') as f_out:
        f_out.attrs['dt'] = dt  # Set the time step attribute
        f_out.create_dataset('rxs/rx1/Ez', data=merged_data_healthy)

    # Draw data with normal plot
    from tools.plot_Bscan import mpl_plot

    rxnumber = 1
    rxcomponent = 'Ez'
    plt = mpl_plot("merged_output_data", merged_data_healthy, dt, rxnumber,
                   rxcomponent)

    # Draw data with gray plot
    file_names = "gray-healthy-" + str(iteration)

    # fig_width = merged_data_healthy.shape[1] / 100
    # # Adjust the size based on the B-scan height
    # fig_height = merged_data_healthy.shape[0] / 100
    fig_width = 15
    fig_height = 15

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.imshow(merged_data_healthy, cmap='gray', aspect='auto')
    plt.axis('off')
    ax.margins(0, 0)  # Remove any extra margins or padding
    fig.tight_layout(pad=0)  # Remove any extra padding

    save_path = current_folder + "/healthy_gray/"
    save_filename = file_names
    plt.savefig(save_path + save_filename + ".png")

    ##########################################################################

    # Cavity:

    with open(file_cavity_1 + '.in', "w") as file:
        file.write('#title: Rotating Straight Scan\n')
        file.write("#domain: {:.3f} {:.3f} 0.002\n".format(
            domain[0], domain[1]))
        file.write("#dx_dy_dz: {} {} {}\n".format(
            resolution, resolution, resolution))
        file.write("#time_window: {}\n".format(time_window))
        file.write()
        file.write('#pml_cells: {} {} 0 {} {} 0\n'.format(
            pml_cells, pml_cells, pml_cells, pml_cells))
        file.write()
        file.write("#geometry_objects_read: {:.3f} {:.3f} 0 h5/defect{}_{}_200.h5 materials.txt\n".format(
            (trunk_center[0]) - radius, (trunk_center[1]) - radius, iteration, current_model_run - 1))
        file.write()
        file.write('#waveform: ricker 1 1e9 my_ricker\n')
        file.write("#hertzian_dipole: z {:.3f} {:.3f} {:.3f} my_ricker\n".format(
            src_position[0], src_position[1], src_position[2]))
        file.write("#rx: {:.3f} {:.3f} {:.3f}".format(
            rx_position[0], rx_position[1], rx_position[2]))
        file.write("#python:")
        file.write("if (current_model_run == (b_scan_cnt - 1)/2):")
        file.write(
            "    print('#geometry_objects_write: 0 0 0 {:.3f} {:.3f} 0.002 new'.format(domain[0], domain[1])')")
        file.write("end_python:")

        api(file_cavity_1 + '.in',
            n=b_scan_cnt,
            geometry_only=False,
            geometry_fixed=False,
            # gpu=[0]
            )
    print('Done B-scan')
    merge_files(file_cavity_1, True)

    # Remove decoupling
    with open('src_only.in', "w") as file:
        file.write('#title: Rotating Straight Scan\n')
        file.write("#domain: {:.3f} {:.3f} 0.002\n".format(
            domain[0], domain[1]))
        file.write("#dx_dy_dz: {} {} {}\n".format(
            resolution, resolution, resolution))
        file.write("#time_window: {}\n".format(time_window))
        file.write('#pml_cells: {} {} 0 {} {} 0\n'.format(
            pml_cells, pml_cells, pml_cells, pml_cells))
        file.write()
        file.write('#waveform: ricker 1 1e9 my_ricker\n')
        file.write("#hertzian_dipole: z {:.3f} {:.3f} {:.3f} my_ricker\n".format(
            src_position[0], src_position[1], src_position[2]))
        file.write("#rx: {:.3f} {:.3f} {:.3f}".format(
            rx_position[0], rx_position[1], rx_position[2]))

    api('src_only.in',
        n=1,
        geometry_only=False,
        geometry_fixed=False,
        # gpu=[0]
        )
    # Merge 2 data parts
    import h5py
    import numpy as np

    filename1 = file_cavity_1 + '_merged.out'
    output_filename = prefix + 'bscan_cavity.out'
    dt = 0

    with h5py.File(filename1, 'r') as f1:
        data1 = f1['rxs']['rx1']['Ez'][()]
        dt = f1.attrs['dt']

    with h5py.File('src_only.out', 'r') as f1:
        data_source = f1['rxs']['rx1']['Ez'][()]

    src = data_source
    src = src[:, np.newaxis]
    Ez0 = np.repeat(src, b_scan_cnt, axis=1)

    merged_data_cavity = np.subtract(data1, Ez0)

    # Create a new output file and write the merged data
    with h5py.File(output_filename, 'w') as f_out:
        f_out.attrs['dt'] = dt  # Set the time step attribute
        f_out.create_dataset('rxs/rx1/Ez', data=merged_data_cavity)

    # Draw data with normal plot
    from tools.plot_Bscan import mpl_plot

    rxnumber = 1
    rxcomponent = 'Ez'
    plt = mpl_plot("merged_output_data", merged_data_cavity, dt, rxnumber,
                   rxcomponent)

    # Draw data with gray plot
    file_names = "gray-cavity-" + str(iteration)

    # fig_width = merged_data_cavity.shape[1] / 100
    # # Adjust the size based on the B-scan height
    # fig_height = merged_data_cavity.shape[0] / 100
    fig_width = 15
    fig_height = 15

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    plt.imshow(merged_data_cavity, cmap='gray', aspect='auto')
    plt.axis('off')
    ax.margins(0, 0)  # Remove any extra margins or padding
    fig.tight_layout(pad=0)  # Remove any extra padding

    save_path = current_folder + "/cavity_gray/"
    save_filename = file_names
    plt.savefig(save_path + save_filename + ".png")

    #############################################################################################
    # Save B-scan and records

    def increment_file_index(string):
        result = ""
        index_start = -1
        index_end = -1

        for i, char in enumerate(string):
            if char.isdigit():
                if index_start == -1:
                    index_start = i
                index_end = i
            elif index_start != -1:
                break

        if index_start == -1:
            return string

        index = int(string[index_start:index_end + 1])
        next_index = index + 1
        replaced_index = str(next_index)
        result = string[:index_start] + replaced_index + string[index_end + 1:]
        return result

    source_file_healthy = prefix + "bscan_healthy.out"
    source_file_cavity = prefix + "bscan_cavity.out"
    destination_file_healthy = current_folder + \
        '/bscan/bscan_healthy_' + str(iteration) + '.out'
    destination_file_cavity = current_folder + \
        '/bscan/bscan_cavity_' + str(iteration) + '.out'

    while os.path.isfile(destination_file_healthy):
        destination_file_healthy = increment_file_index(
            destination_file_healthy)

    while os.path.isfile(destination_file_cavity):
        destination_file_cavity = increment_file_index(destination_file_cavity)

    with open(source_file_healthy,
              "rb") as source, open(destination_file_healthy,
                                    "wb") as destination:
        destination.write(source.read())

    with open(source_file_cavity, "rb") as source, open(destination_file_cavity,
                                                        "wb") as destination:
        destination.write(source.read())

    print('B_scan_healthy exported to file: {}'.format(destination_file_healthy))
    print('B_scan_cavity exported to file: {}'.format(destination_file_cavity))

    # Plot the cavity_only signal
    import matplotlib.pyplot as plt
    bscan_data = np.abs(merged_data_cavity - merged_data_healthy)
    # Plot the cavity_only signal
    # log_data = np.log(1 + bscan_data)

    # # Normalize the logarithmic data between 0 and 1
    # log_data_normalized = (log_data - np.min(log_data)) / \
    #     (np.max(log_data) - np.min(log_data))

    # Set the figure size
    # Adjust the size based on the B-scan width
    fig_width = 15
    fig_height = 15

    # Create the figure and axes objects with the desired size and aspect ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Generate the image plot with the logarithmic-scaled and normalized data
    im = ax.imshow(bscan_data, cmap='gray', aspect='auto')
    ax.axis('off')
    ax.margins(0, 0)  # Remove any extra margins or padding
    fig.tight_layout(pad=0)  # Remove any extra padding

    save_path = current_folder + "/cavity_only_gray/"
    save_filename = "cavity-only_" + str(iteration)
    plt.savefig(save_path + save_filename + ".png")

    # Export to data.csv
    csv_file = './output/csv/data.csv'
    # Prepare the data to be written to the CSV file
    data = [
        iteration, radius, src_to_trunk, src_to_rx, eps_trunk,
        time_window, pml_cells, x_gap, y_gap, src_to_pml, domain[0],
        domain[1], domain[2]
    ]

    # Write the data to the CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    print("Data appended to the CSV file.")
