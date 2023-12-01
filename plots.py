import json
import math
from tkinter import Tk, filedialog
import numpy as np
#from ahrs import filters
import ahrs as ahrs
from ahrs.filters import Madgwick
from ahrs.filters import EKF
from ahrs.common.orientation import q2R, q2euler
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rr
from scipy import integrate as dx
# Open file dialog to choose acceleration file (.json)
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

root.destroy()
# Load accelerometer data
with open(file_path) as f:
    data = json.load(f)
    # Assuming your JSON structure contains 'x' and 'y' arrays, modify as needed
accx = data['accX']
accy = data['accY']
accz = data['accZ']
xgyro = data['magX']
ygyro = data['magY']
zgyro = data['magZ']
timestampsAcc = data['timeStampsAcc']
timestampsGyro = data['timeStampsGyro']
plt.figure(figsize=(10, 6))

# Plotting accX
plt.subplot(3, 1, 1)  # Creating subplot for accX
plt.plot(timestampsAcc, accx, label='accX')
plt.xlabel('Timestamps')
plt.ylabel('Acceleration (accX)')
plt.title('Acceleration X')
plt.legend()

# Plotting accY
plt.subplot(3, 1, 2)  # Creating subplot for accY
plt.plot(timestampsAcc, accy, label='accY')
plt.xlabel('Timestamps')
plt.ylabel('Acceleration (accY)')
plt.title('Acceleration Y')
plt.legend()

# Plotting accZ
plt.subplot(3, 1, 3)  # Creating subplot for accZ
plt.plot(timestampsAcc, accz, label='accZ')
plt.xlabel('Timestamps')
plt.ylabel('Acceleration (accZ)')
plt.title('Acceleration Z')
plt.legend()

plt.tight_layout()  # Adjusting layout for subplots
plt.show()