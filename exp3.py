import sys
import numpy as np

GROUP = ['habitation', 'selective_logging', 'slash_burn']

labels = GROUP

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

print label_map
print inv_label_map

