import numpy as np
# Important: Change it to TACC machine data location
from tqdm import tqdm
import pickle
import os
from pymatgen.ext.matproj import MPRester


if __name__ == '__main__':
    data_location = './data'
    material_id = np.array(pickle.load(open(os.path.join(data_location,'id.txt'),'rb')))
    n = len(material_id)
    nrf = np.zeros((n,6))

    with MPRester("GukXVgPTxL4ABdsU") as m:
        for i in tqdm(range(n)):
            try:
                structure = m.get_structure_by_material_id(material_id[i])
                nrf[i,0] = structure.lattice.a
                nrf[i,1] = structure.lattice.b
                nrf[i,2] = structure.lattice.c
                nrf[i,3] = structure.lattice.alpha
                nrf[i,4] = structure.lattice.beta
                nrf[i,5] = structure.lattice.gamma
            except:
                print(material_id[i])
    with open('params.npy', 'wb') as f:
        np.save(f, nrf)
            
