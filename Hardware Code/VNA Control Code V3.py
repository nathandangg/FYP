import time
import pyvisa
import math
import serial
import numpy as np
import csv
from itertools import zip_longest
###################################################################################################
# User Parameters
addr = 'TCPIP0::localhost::hislip_PXI10_CHASSIS1_SLOT1_INDEX0::INSTR' # Device VISA address
VNA_model = 'P5021A'
COM_port = 'COM7' #Arduino
test = False #If not using slider then True
tolerances = [99, 99, 99, 99] #Tr1 to 4
max_attempts=20 # max_attempt for each ascan 
scan_length = 100
trunk_distance = 35
b_scan_count = 51
conversion = [50, 847/180]#from cm, degrees to motor steps
continue_from = 75

###################################################################################################
# Define Helper Function
def capture_and_process_data(vna, numtrace=4):
    trace_indices = ",".join(str(i) for i in range(1, numtrace + 1))
    vna.write("SENS:SWE:MODE SING")
    vna.query("*OPC?")

    query_string = f'CALC:DATA:MSD? "{trace_indices}"'
    data = vna.query_binary_values(query_string, datatype='d')

    traces = [data[len(data)//numtrace*i:len(data)//numtrace*(i+1)] for i in range(numtrace)]
    real_parts = np.array([trace[::2] for trace in traces])
    imag_parts = np.array([trace[1::2] for trace in traces])

    magnitudes = np.array([np.abs(np.array(real) + 1j * np.array(imag)) for real, imag in zip(real_parts, imag_parts)])

    return real_parts, imag_parts, magnitudes

def calculate_relative_error_percent(previous_magnitudes, magnitudes):
    re_percentages = []
    for prev_trace, current_trace in zip(previous_magnitudes, magnitudes):
        re = np.abs(current_trace - prev_trace)/np.array(current_trace)
        re_percentage = np.mean(re)*100  # Calculate RMSE as a percentage
        re_percentages.append(re_percentage)
    return re_percentages

def calculate_rmse_percent(previous_magnitudes, magnitudes):
    rmse_percentages = []
    for prev_trace, current_trace in zip(previous_magnitudes, magnitudes):
        rmse = np.average(np.square(current_trace - prev_trace))
        rmse_percentage = (rmse/np.mean(current_trace))*100  # Calculate RMSE as a percentage
        rmse_percentages.append(rmse_percentage)
    return rmse_percentages

# 2 RUN ONLY VERSION
def run_until_convergence(vna, numtrace=4, tolerances=None, max_attempts=20):
    if tolerances is None:
        tolerances = [1] * numtrace  # Default tolerance as a percentage for all traces

    previous_magnitudes = None
    previous_real = None
    previous_imag = None

    for attempt in range(max_attempts):
        print(f"Starting Attempt: {attempt + 1}")
        real, imag, magnitudes = capture_and_process_data(vna, numtrace)

        if previous_magnitudes is not None:
            # Calculate RMSE percentages for each trace
            e_percentages = calculate_relative_error_percent(previous_magnitudes, magnitudes)

            # Print the RMSE percentage for each trace
            for i, rmse_percentage in enumerate(e_percentages):
                print(f"Trace {i + 1}: Error% = {rmse_percentage:.2f}%")

            # Check if each trace has converged within its respective tolerance
            converged = all(rmse <= tol for rmse, tol in zip(e_percentages, tolerances))

            if converged:
                print(f"Converged after {attempt + 1} attempts.")
                # Calculate the mean of the current and previous real and imaginary data
                mean_real = [(r + prev_r)/2 for r, prev_r in zip(real, previous_real)]
                mean_imag = [(i + prev_i)/2 for i, prev_i in zip(imag, previous_imag)]
                return mean_real, mean_imag

        # If not converged, print a message and continue
        if attempt > 0:
            print(f"Attempt {attempt + 1}: Magnitudes not converged.")
        
        # Update the previous to the current values
        previous_magnitudes = magnitudes
        previous_real = real
        previous_imag = imag

    print("Maximum attempts reached. Convergence not achieved.")
    return None, None

def connect_to_vna(addr, VNA_model):
    try:
        print('\nAttempting to connect to: ' + addr)

        # Opening VISA resource manager:
        rm = pyvisa.ResourceManager()

        # Connect to equipment at my address 'addr' above
        vna = rm.open_resource(addr, timeout=5000)

        # Check what the device at this VISA address responds as:
        resp = vna.query('*IDN?')

        if VNA_model in resp:
            print('\nSuccessfully connected to Keysight VNA simulator!\n')
            # Set format of instrument to return
            vna.write('FORMat REAL,64')

            # Set byte order to swapped (little-endian) format
            vna.write('FORMat:BORDer SWAP')

            # Set starting condition to HOLD
            vna.write("SENS:SWE:MODE HOLD")
            time.sleep(0.1)

            return vna
        else:
            print('\nUnable to connect to Keysight VNA! Check that the correct VNA model is chosen.\n')
            return None

    except pyvisa.errors.VisaIOError as e:
        print(f'Error connecting to VNA: {e}')
        return None

def connect_to_arduino(comport):
    arduino = serial.Serial(comport)
    arduino.baudrate = 9600
    arduino.timeout = 5
    arduino.bytesize = 8
    arduino.parity = 'N'
    arduino.stopbits = 1
    print("Connected to Arduino via ", comport)
    return arduino

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

    return current_angle

def bscan(L, H, count):
    mid = L/2
    step = (L/(count-1))
    dist = []
    angle = []
    for i in range(count):
        dist.append(i*step)
        angle.append(find_angle(i*step,mid,H))
    dist = np.array(dist)

    return dist,angle
###################################################################################################
# Main Function
if not test:
    arduino = connect_to_arduino(COM_port)

vna = connect_to_vna(addr, VNA_model)
pos, rot = bscan(scan_length, trunk_distance, b_scan_count)

if vna is not None:
    input("Press Enter to start the program...")
    start = time.time()
    i = round((continue_from - 1)/2)

    # Start the program
    print(f'Moving to dist: {pos[i]:.2f} and angle: {rot[i]:.2f}')
    if not test:
        # arduino.write(b'a')
        arduino.write(f'P {round(pos[i]*conversion[0])} {round(rot[i]*conversion[1])}'.encode("utf-8"))

    # Main Loop
    while True:
        # Get the start A-scan signal
        if not test:
            msg = arduino.read()
        else:
            msg = b'b'
        
        # Check if the message is equal to b'b', add condition for i == 0 for old code
        if msg == b'b':
            print(f"\nInterval: {(2 * i + 1)}")
            starttime = time.time()

            # Sweeping
            real, imag = run_until_convergence(vna, tolerances = tolerances, max_attempts = max_attempts)
            if(real is None or imag is None):
                break

            endtime = time.time()
            print("\nTime taken for data to stablise:%.2f seconds" %(endtime - starttime))

            #write data into csv
            data = [x for pair in zip(real, imag) for x in pair]
            export_data = zip_longest(*data, fillvalue='')
            with open(f'{2 * i + 1}.csv', 'w', encoding="ISO-8859-1", newline='') as file:
                write = csv.writer(file)
                write.writerows(export_data)
            file.close()

            # End Condition
            if i >= (b_scan_count-1):
                break

            # Next A-scan trace
            i = i + 1

            # Move the slider to the next position
            if not test:
                # arduino.write(b'a')
                print(f'\nMoving to Dist: {pos[i]:.2f} and Angle: {rot[i]:.2f}')
                arduino.write(f'P {round(pos[i]*conversion[0])} {round(rot[i]*conversion[1])}'.encode("utf-8"))
                time.sleep(0.1)
            

    end = time.time()
    print(f'\nTime taken for entire bscan {end - start:.2f}')
    
    # Reset Arduino
    if not test:
        print('\nEnding Program, Resetting Arduino ...')
        arduino.write(b'c')
        time.sleep(0.1)
        arduino.close()
