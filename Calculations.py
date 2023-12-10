# Open file dialog to choose acceleration file (.json)
import json
from tkinter import Tk, filedialog

import numpy as np
from scipy import integrate as dx
import math
import matplotlib

matplotlib.use('TkAgg')  # or another backend that works for you
import matplotlib.pyplot as plt

# Define the EWMA filter function
def ewma_filter(alpha, x, y_prev):
    return alpha * x + (1 - alpha) * y_prev

# Define the complementary filter function
def complementary_filter(alpha, x1, x2):
    return alpha * x1 + (1 - alpha) * x2

root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

root.destroy()
# Load accelerometer data
with open(file_path) as f:
    data = json.load(f)

# Filter out zero values
non_zero_indices = np.where(np.array(data['accX']) != 0)[0]

x = np.array(data['accX'])[non_zero_indices]
y = np.array(data['accY'])[non_zero_indices]
z = np.array(data['accZ'])[non_zero_indices]
xgyro = np.array(data['magX'])[non_zero_indices]
ygyro = np.array(data['magY'])[non_zero_indices]
zgyro = np.array(data['magZ'])[non_zero_indices]
timestampsAcc = data['timeStampsAcc']
timestampsGyro = data['timeStampsGyro']

xthreshholded = []
ythreshholded = []
zthreshholded = []

# Plot raw X-axis acceleration data as a subplot
plt.plot(x, color='red', label='Raw X')
plt.plot(y, color='green', label='Raw Y')
plt.plot(z, color='blue', label='Raw Z')
plt.legend()
plt.grid(True)
plt.show()

# Linear Acceleration with EWMA filter
alpha_ewma = 0.2  # Adjust this value based on your requirements (between 0 and 1)
filteredEWMA_x = [x[0]]  # Initialize the first value
filteredEWMA_y = [x[0]]
filteredEWMA_z = [x[0]]

for i in range(1, len(x)):
    filtered_value = ewma_filter(alpha_ewma, x[i], filteredEWMA_x[i - 1])
    filteredEWMA_x.append(filtered_value)
for i in range(1, len(y)):
    filtered_value = ewma_filter(alpha_ewma, y[i], filteredEWMA_y[i - 1])
    filteredEWMA_y.append(filtered_value)
for i in range(1, len(z)):
    filtered_value = ewma_filter(alpha_ewma, z[i], filteredEWMA_z[i - 1])
    filteredEWMA_z.append(filtered_value)

# Plot raw X-axis acceleration data and filtered data
plt.plot(x, color='red', label='Raw X')
plt.plot(y, color='green', label='Raw Y')
plt.plot(z, color='blue', label='Raw Z')
plt.plot(filteredEWMA_x, color='orange', label=f'Filtered X (EWMA, alpha={alpha_ewma})')
plt.plot(filteredEWMA_y, color='orange', label=f'Filtered Y (EWMA, alpha={alpha_ewma})')
plt.plot(filteredEWMA_z, color='orange', label=f'Filtered Z (EWMA, alpha={alpha_ewma})')
plt.legend()
plt.grid(True)
plt.show()

# Algorithm 2 - Sensor Fusion with Complementary Filter
alpha_complementary = 0.7  # Adjust this value based on your requirements (between 0 and 1)
linear_acceleration = np.array([math.sqrt(xi**2 + yi**2 + zi**2) - 9.82 for xi, yi, zi in zip(x, y, z)])
angular_input = np.array([math.sqrt(xi**2 + yi**2 + zi**2) for xi, yi, zi in zip(xgyro, ygyro, zgyro)])

filtered_linear_acceleration = [linear_acceleration[0]]  # Initialize the first value

for i in range(1, len(linear_acceleration)):
    filtered_value = complementary_filter(alpha_complementary, linear_acceleration[i], angular_input[i])
    filtered_linear_acceleration.append(filtered_value)

# Plot raw linear acceleration and filtered data
plt.plot(linear_acceleration, color='orange', label='Raw Linear Acceleration')
plt.plot(filtered_linear_acceleration, color='blue', label=f'Filtered Linear Acceleration (Complementary, alpha={alpha_complementary})')
plt.legend()
plt.grid(True)
plt.show()


totalAcc = np.zeros(len(x))
totalAccFilteredEWMA = np.zeros(len(x))
g = 9.82
time_step = 1/52
check = False
for i in range(len(x)):
    if x[i] != 0:
        totalAcc[i] = -math.sqrt(x[i]**2 + y[i]**2 + z[i]**2) + 9.82
        if totalAcc[i] > -0.8 and check != True:
            totalAcc[i] = 0
        elif totalAcc[i] < -0.8:
            check = True


totalAccFilteredEWMA[i] = -math.sqrt(filteredEWMA_x[i] ** 2 + filteredEWMA_y[i] ** 2 + filteredEWMA_z[i] ** 2) + g

print(totalAcc)
velocityTot = dx.cumtrapz(totalAcc, dx=time_step)
velocityTotFilteredEWMA = dx.cumtrapz(totalAccFilteredEWMA, dx=time_step)
velocityTotFilteredComplimentary = dx.cumtrapz(filtered_linear_acceleration, dx=time_step)

filteredEWMA_totalAcc = [totalAcc[0]]
for i in range(1, len(totalAcc)):
    filtered_value = ewma_filter(alpha_ewma, totalAcc[i], filteredEWMA_totalAcc[i - 1])
    filteredEWMA_totalAcc.append(filtered_value)

velocityTotFilteredEWMA2 = dx.cumtrapz(filteredEWMA_totalAcc, dx=time_step)

plt.title('Raw total Acceleration vs EWMA filtered total Acceleration vs Complimentary filter')
plt.plot(totalAcc, color='red', label='Raw Acceleration')
plt.plot(filteredEWMA_totalAcc, color='blue', label='EWMA filtered total Acceleration')
plt.plot(filtered_linear_acceleration, color='green', label='Complimentary filtered total Acceleration')
plt.legend()
plt.grid(True)
plt.show()

plt.title('Velocity raw vs Velocity EWMA vs Velocity Complimentary')
plt.plot(velocityTot, color='red', label='Raw Velocity')
plt.plot(velocityTotFilteredEWMA2, color='green', label='EWMA Velocity')
#plt.plot(velocityTotFilteredComplimentary, color='blue', label='Complimentary Velocity')
plt.legend()
plt.grid(True)
plt.show()

# Constants
STATIC_SAMPLES_THRESHOLD = 10
STATIC_ACCELERATION_THRESHOLD = 0.7
samples_count = 0
SaveIndexes = []
for i in range(len(x)):
    # Check if all axes are below the threshold
    if (
            abs(x[i]) <= STATIC_ACCELERATION_THRESHOLD
            and abs(y[i]+9.82) <= STATIC_ACCELERATION_THRESHOLD
            and abs(z[i]) <= STATIC_ACCELERATION_THRESHOLD
    ):
        samples_count += 1
    else:
        samples_count = 0

    # Check if the stationary condition is met
    if samples_count >= STATIC_SAMPLES_THRESHOLD:
        print("Device is stationary")
        SaveIndexes.append(i)
    else:
        print("Device is not stationary")


for index in range(len(SaveIndexes)):
    velocityTot[SaveIndexes[index]] = 0


print(SaveIndexes)

plt.plot(velocityTot)
plt.show()

