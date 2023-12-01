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