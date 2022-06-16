import os
import glob
import json
import csv
import pickle
import sys
from tqdm import tqdm
rootdir = '/mnt/data/datasets/Material_Raw_Data'
material = {}

# for each_file in tqdm(glob.glob(rootdir + '/data_*/mp-*/xrd.csv')):
#     idx = each_file.split('/')[-2]
#     with open(each_file, newline='') as f:
#         reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
#         data = list(reader)
#     # extract xrd (only non-zero rows)
#     material[idx] = {}
#     material[idx]['xrd'] = [row for row in data if any(row)]

for each_file in tqdm(glob.glob(rootdir + '/data_*/mp-*/struct.json')):
    idx = each_file.split('/')[-2]
    material[idx] = {}
    with open(each_file) as f:
        data = json.load(f)
        # parse file
        obj = json.loads(data)
        # extract lattice parameters
        lattice = obj['lattice']
        a = float(lattice['a'])
        b = float(lattice['b'])
        c = float(lattice['c'])
        alpha = float(lattice['alpha'])
        beta = float(lattice['beta'])
        gamma = float(lattice['gamma'])
        material[idx]['sites'] = obj['sites']
        # save parameters
    material[idx]['param'] = [a, b, c, alpha, beta, gamma]
    

for each_file in tqdm(glob.glob(rootdir + '/data_*/mp-*/energy.json')):
    idx = each_file.split('/')[-2]
    with open(each_file) as f:
        data = json.load(f)
        # parse file
        obj = json.loads(data)
        # extract lattice parameters
        coh_eng = float(obj['cohesive_energy_per_atom'])
        # save energy
    material[idx]['energy'] = coh_eng

with open('material_sites_no_xrd.pkl', 'wb') as f:
    pickle.dump(material, f)
