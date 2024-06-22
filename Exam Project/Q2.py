from types import SimpleNamespace
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

class CareerClass:

    def __init__(self):
        """ initialize the Career class with default parameters """
        par = self.par = SimpleNamespace()