import json
from PIL import Image
import pandas as pd
import numpy as np
import sys
import time
import plots


plots.plot_curve(values=np.random.random(300), title='Training Loss', file_name='_tr_loss.jpg')