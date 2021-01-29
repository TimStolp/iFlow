import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', default=None, help='path to data file in .txt format. (default None)')

args = parser.parse_args()

with open(args.file, 'r') as f:
    coeffs = f.readline().split(';')[1:]

array = np.array([c.strip('[]').split() for c in coeffs]).astype('float')
std = array.std(axis=0)
print('standard deviations: ', std)