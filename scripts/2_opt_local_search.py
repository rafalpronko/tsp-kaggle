import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import random
import numpy as np
from sympy import primerange

cities = pd.read_csv("data/cities.csv")

tour = pd.read_csv("submit/submition_181206_v_1516768.4970248034.csv")['Path'].tolist()

primes = list(primerange(0, len(cities)))

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

def local2op(path, i, k):
    tmp = path[i:k]
    return path[0:i] + tmp[::-1] + path[k:]

best_tour = score_tour(tour)
max_number = len(tour) - 1
tour_best = tour.copy()

for i in range(1,max_number):
    if i%10000 == 0:
        print("Left: {}".format(max_number - i))
    for k in range(i+1, i+10):
        if k - i > 1:
            fliped_tour = local2op(tour_best.copy(), i, k)
            fitness_tmp = score_tour(fliped_tour)
            if fitness_tmp < best_tour:
                print(fitness_tmp)
                best_tour = fitness_tmp
                pd.DataFrame({"Path": fliped_tour}).to_csv("submit/submition_181208_v_{}.csv".format(fitness_tmp), index=False)
                tour_best = fliped_tour.copy()

