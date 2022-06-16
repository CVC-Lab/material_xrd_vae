import os
import json
import csv
import pickle
import sys
rootdir = '/mnt/data/datasets/Material_Raw_Data'
id = []
xrd = []
params = []
energy = []
for subdir, dirs, files in os.walk(rootdir):
    #####ID#####
    for dir in dirs:
        if 'mp' in dir:
            id.append(dir)
    #####XRD#####
    # read file
    for file in files:
        if file == 'xrd.csv':
            with open(os.path.join(subdir, file), newline='') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                data = list(reader)
            # extract xrd (only non-zero rows)
            xrd.append([row for row in data if any(row)]) 
    #####PARAMETERS#####
    # read file
    for file in files:
        if file == 'struct.json':
            with open(os.path.join(subdir, file)) as f:
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
            # save parameters
            params.extend([[a, b, c, alpha, beta, gamma]])
    #####ENERGY#####
    # read file
    for file in files:
        if file == 'energy.json':
            with open(os.path.join(subdir, file)) as f:
                data = json.load(f)
            # parse file
            obj = json.loads(data)
            # extract lattice parameters
            coh_eng = float(obj['cohesive_energy_per_atom'])
            # save energy
            energy.append(coh_eng)
# create files
with open('id.txt', 'wb') as f:
    pickle.dump(id, f)
with open('xrd.txt', 'wb') as f:
    pickle.dump(xrd, f)
with open('params.txt', 'wb') as f:
    pickle.dump(params, f)
with open('energy.txt', 'wb') as f:
    pickle.dump(energy, f)
