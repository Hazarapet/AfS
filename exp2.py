import cv2
import json
import numpy as np
import utils.common as common_util
from PIL import Image
import matplotlib.pyplot as plt
import theano.tensor as T

a = T.scalar('a')

print T.round(0.5).eval()
print T.clip(0.2, 0, 1).eval()