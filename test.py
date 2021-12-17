from multiprocessing import Pool
import multiprocessing
import numpy as np
import time
import math

def main():
    pool = Pool()
    out = pool.map(rand, range(10))
    print(multiprocessing.cpu_count())

def rand(num):
    np.random.seed(int(time.time() + math.sin(num) * 1000))
    return np.random.choice(10)

if __name__ == "__main__":
    main()