import csv
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import argparse

parser = argparse.ArgumentParser(description='Script description')

# Add an argument for starting_index
parser.add_argument('--start', type=int, help='Starting index value')

# Add an argument for iteration_cnt
parser.add_argument('--count', type=int, help='Iteration count value')

args = parser.parse_args()

# Access the values of starting_index and iteration_cnt
starting_index = args.start
iteration_cnt = args.count


def EnclosedCurve(r_random, N_angle, H, Center_X, Center_Y):
    ang = np.linspace(0, 2*np.pi/N_angle*(N_angle-1), N_angle)

    rho = np.random.rand(H) * 0.15 * np.logspace(-1.5, -2.5, H)
    phi = np.random.rand(H) * 2 * np.pi

    r = np.zeros(N_angle) + r_random

    for h in range(H):
        r = r + rho[h] * np.sin(h * ang + phi[h])

    x = r * np.cos(ang) + Center_X
    y = r * np.sin(ang) + Center_Y

    return x, y, r


def generate_random_array(N, mean, stddev, resolution, min_value, max_value):
    random_array = []
    while len(random_array) < N:
        random_value = rd.normalvariate(mean, stddev)
        rounded_value = round(random_value / resolution) * resolution
        if min_value <= rounded_value <= max_value:
            random_array.append(rounded_value)
    return random_array


def save_image(filename, x_trunk_L1, y_trunk_L1, x_trunk_L2, y_trunk_L2, x_trunk_L3, y_trunk_L3, x_Bcavity, y_Bcavity, layer1, layer2, layer3, cavity_colour):
    plt.figure(figsize=(10, 10))
    plt.fill(x_trunk_L1, y_trunk_L1, color=layer1, linestyle='none')
    # plt.fill(x_trunk_L2, y_trunk_L2, color=layer2, linestyle='none')
    # plt.fill(x_trunk_L3, y_trunk_L3, color=layer3, linestyle='none')
    plt.fill(x_Bcavity, y_Bcavity, color=cavity_colour, linestyle='none')
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.axis([-0.35, 0.35, -0.35, 0.35])
    plt.axis('off')
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()


# N_trunk = 10
N_trunk = iteration_cnt

# Set parameters
Rmin_trunk = 0.15
Rmax_trunk = 0.30
CenTrunk_X = 0
CenTrunk_Y = 0
N_angle = 360
safety_d = 0.01

# Small cavity, if not needed can be removed
Rmin_Scavity = 0.03
Rmax_Scavity = 0.09

layer1 = [1, 1, 0]
layer2 = [1, 0.8, 0]
layer3 = [1, 0.6, 0]
cavity_colour = [1, 0.2, 0]

# Initialize arrays to store data
x_trunk_L1 = np.zeros((N_trunk, N_angle))
y_trunk_L1 = np.zeros((N_trunk, N_angle))
x_trunk_L2 = np.zeros((N_trunk, N_angle))
y_trunk_L2 = np.zeros((N_trunk, N_angle))
x_trunk_L3 = np.zeros((N_trunk, N_angle))
y_trunk_L3 = np.zeros((N_trunk, N_angle))
x_Bcavity = np.zeros((N_trunk, N_angle))
y_Bcavity = np.zeros((N_trunk, N_angle))
r_trunk_L3 = np.zeros((N_trunk, N_angle))
r_trunk_L2 = np.zeros((N_trunk, N_angle))
r_trunk_L1 = np.zeros((N_trunk, N_angle))
r_Bcavity = np.zeros((N_trunk, N_angle))
hole_factor = np.zeros(N_trunk)
R_center_Bcavity = np.zeros(N_trunk)
theta_Bcavity = np.zeros(N_trunk)
CenBcavity_X = np.zeros(N_trunk)
CenBcavity_Y = np.zeros(N_trunk)

# Up: upperbound, IS: conductive layer, HW: heartwood
up_IS = 0.9
lo_IS = 0.75
up_bark = 1.1
lo_bark = 1.05

radius = generate_random_array(N_trunk, 0.3, 0.06, 0.002, 0.2, 0.4)
eps_trunk = generate_random_array(N_trunk, 10, 2, 0.1, 5, 15)
eps_cavity = 1
# eps_cavity = generate_random_array(N_trunk, 10, 2, 0.1, 5, 15)

for i in range(N_trunk):
    H = 8
    R_rand_trunk = np.random.rand(1)*(Rmax_trunk-Rmin_trunk) + Rmin_trunk
    x_trunk_L2[i], y_trunk_L2[i], r_trunk_L2[i] = EnclosedCurve(
        R_rand_trunk, N_angle, 13, CenTrunk_X, CenTrunk_Y)

    ratio_L3 = np.random.rand()*(up_IS-lo_IS) + lo_IS
    x_trunk_L3[i] = ratio_L3 * x_trunk_L2[i]
    y_trunk_L3[i] = ratio_L3 * y_trunk_L2[i]
    r_trunk_L3[i] = ratio_L3 * r_trunk_L2[i]

    ratio_L1 = np.random.rand()*(up_bark-lo_bark) + lo_bark
    x_trunk_L1[i] = ratio_L1 * x_trunk_L2[i]
    y_trunk_L1[i] = ratio_L1 * y_trunk_L2[i]
    r_trunk_L1[i] = ratio_L1 * r_trunk_L2[i]

    Rmin_trunk_L3 = np.min(r_trunk_L3[i])
    Rmax_Bcavity = 0.8 * Rmin_trunk_L3
    Rmin_Bcavity = 0.05
    hole_factor[i] = np.random.rand(1)
    R_rand_Bcavity = hole_factor[i] * \
        (Rmax_Bcavity - Rmin_Bcavity) + Rmin_Bcavity

    min_d_Bcavity = 0.02
    max_d_Bcavity = Rmin_trunk_L3 - R_rand_Bcavity - safety_d
    R_center_Bcavity[i] = min_d_Bcavity + \
        np.random.rand() * (max_d_Bcavity - min_d_Bcavity)
    theta_Bcavity[i] = 2 * np.pi * np.random.rand()
    CenBcavity_X[i] = R_center_Bcavity[i] * np.cos(theta_Bcavity[i])
    CenBcavity_Y[i] = R_center_Bcavity[i] * np.sin(theta_Bcavity[i])

    shape_factor = np.random.rand()
    max_shape = 20
    min_shape = 15
    shape_cavity = round(shape_factor * (max_shape - min_shape) + min_shape)
    x_Bcavity[i], y_Bcavity[i], r_Bcavity[i] = EnclosedCurve(
        R_rand_Bcavity, N_angle, shape_cavity, CenBcavity_X[i], CenBcavity_Y[i])

    save_image('image/defect/defect{}.png'.format(i+starting_index), x_trunk_L1[i], y_trunk_L1[i], x_trunk_L2[i], y_trunk_L2[i],
               x_trunk_L3[i], y_trunk_L3[i], x_Bcavity[i], y_Bcavity[i], layer1, layer2, layer3, cavity_colour)
    save_image('image/healthy/healthy{}.png'.format(i+starting_index), x_trunk_L1[i], y_trunk_L1[i], x_trunk_L2[i], y_trunk_L2[i],
               x_trunk_L3[i], y_trunk_L3[i], np.zeros((1, N_angle)), np.zeros((1, N_angle)), layer1, layer2, layer3, cavity_colour)

    print(CenBcavity_X[i], CenBcavity_Y[i], np.max(
        r_Bcavity[i]), np.max(r_trunk_L1[i]), radius[i], eps_trunk[i])


def max_row(arr):
    # Use numpy's max function along axis 1 to get the maximum of each row
    max_values = np.max(arr, axis=1)
    return max_values


def save_arrays_to_csv(file_path, *arrays):
    # Transpose the arrays to create rows from the corresponding elements
    rows = zip(*arrays)

    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write each row to the CSV file
        for row in rows:
            writer.writerow(row)


# Example usage:
# Assuming you have six 1D arrays: array1, array2, array3, array4, array5, array6
# Specify the file path where you want to save the CSV file
file_path = 'data-config0.csv'

# Call the function to save the arrays to the CSV file (it will overwrite the file if it exists)
save_arrays_to_csv(file_path, CenBcavity_X, CenBcavity_Y, max_row(
    r_Bcavity), max_row(r_trunk_L1), radius, eps_trunk)
