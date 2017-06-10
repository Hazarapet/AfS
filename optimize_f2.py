import numpy as np
import utils.common as common

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = common.f2_score(y, p2).eval()
    return score

  x = [0.2] * 17

  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      print 'score: ', score, ', best_score: ', best_score
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x