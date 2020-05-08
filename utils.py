import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

def load_material_data(data_location):

    data = sio.loadmat(data_location)

    input_mat = data['MP']

    # count data in different classes
    id = input_mat[:,0]
    atom_type = input_mat[:,1]
    energy = input_mat[:,2] # target value
    X = input_mat[:,3:] # training data

    return X,id,atom_type,energy

    

def load_material_data_train_test_split(data_location, return_energy=False):
    X,_,atom_type, e = load_material_data(data_location)
    y = np.array(atom_type - 1, dtype=int)
    for i in range(7):
        cnt = np.count_nonzero(atom_type == (i+1))
        #print("Type %d : %d" % (i+1, cnt))
    #print(np.max(y),np.min(y))

    # First train everything
    if return_energy:
        X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(X, y, e, test_size=0.20, random_state=9)
        return X_train, X_test, y_train, y_test, e_train, e_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)
        return X_train, X_test, y_train, y_test


def safe_log(z):
    return torch.log(z + 1e-7)


def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)