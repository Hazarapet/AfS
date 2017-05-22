import sys
import numpy as np
import keras.backend as K
from utils import common as common_util

GROUP = ['artisinal_mine'
         'bare_ground',
         'blow_down',
         'conventional_mine',
         'cultivation',
         'haze',
         'selective_logging']

k = 'haze'

print k in GROUP

