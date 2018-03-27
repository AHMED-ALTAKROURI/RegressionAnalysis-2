import matplotlib.pyplot as plt;

plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6','F7','F8'
          , 'F9','F10','F11','F12','F13','F14','F15','F16'
           ,'F17','F18','F19','F20','F21','F22')
y_pos = np.arange(len(objects))
performance = [52461, 296, 33, 1504, 46524, 26851,27974,77208,27742,77399,33,33,33,61277,74405,3665,77364,51940,59854,38,35,145]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Missing Values')
plt.title('Number of missing values for each feature component')

plt.show()
