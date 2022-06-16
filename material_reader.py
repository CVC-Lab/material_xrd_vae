import pickle
import os
import numpy as np

with open('material_sites_no_xrd.pkl', 'rb') as f:
    material = pickle.load(f)
print (material['mp-9994'])
with open('params.npy', 'rb') as f:
    nrf_mp_website = np.load(f)
            
data_location = './data'
material_id = np.array(pickle.load(open(os.path.join(data_location,'id.txt'),'rb')))

for i in range(len(material_id)):
    material[material_id[i]]['sites']
# for i in range(len(material_id)):
#     print("Sanity check: id = {}, recorded parameter = {}, website parameter = {}".format(material_id[i], material[material_id[i]].get('param','None'), nrf_mp_website[i,:]))