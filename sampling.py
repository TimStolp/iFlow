from scipy import special
import math
import numpy as np


def sampling(xi, eta, p, norm):
    sqrtxi = pow(abs(xi), 0.5)
    num = (2 * p * sqrtxi * norm)
    den = (math.sqrt(math.pi) * math.exp((pow((-eta), 2) / (4 * xi))))
    # z = 1/sqrtxi * -1j * special.erfinv(1j * (num / den)) - (eta / (2 * xi))
    z = 1 / sqrtxi * special.erfinv(num / den) - (eta / (2 * xi))
    return z


if __name__ == '__main__':
    p = np.random.uniform(0, 1)
    z = sampling(-1, 1, p, 1)
    print(z)
