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

    

def load_material_data_train_test_split(data_location):
    X,_,atom_type,_ = load_material_data(data_location)
    y = atom_type - 1
    for i in range(7):
        cnt = np.count_nonzero(atom_type == (i+1))
        print("Type %d : %d" % (i+1, cnt))
    print(np.max(y),np.min(y))

    # First train everything
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

    return X_train, X_test, y_train, y_test