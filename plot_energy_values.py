import os
import numpy as np
import matplotlib.pyplot as plt

energy_values = []
for directory in os.listdir('experiments'):
    npz = np.load(os.path.join('experiments', directory, 'log', 'data', '1.npz'))
    # 'log_normalizer', 'neg_log_det', 'neg_trace', 'loss', 'perf'
    energy_values.append(-npz['loss'][-1])

np.save(os.path.join('results', 'energy_values.npy'), np.array(energy_values))
plt.plot(energy_values, '-.', color='gold', label=f'iFlow:{round(np.mean(energy_values), 4)}')
plt.legend()
plt.xlabel('Seed Number')
plt.ylabel('Energy Value')
plt.show()