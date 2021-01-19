import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

with open(osp.join('results', 'eval_performances.txt')) as f:
    performances = list(map(eval, f.readline().split(',')[1:]))

plt.plot(performances, '-.', color='gold', label=f'iFlow:{round(np.mean(performances), 4)}({round(np.std(performances), 4)})')
plt.legend()
plt.xlabel('Seed Number')
plt.ylabel('MCC')
plt.show()