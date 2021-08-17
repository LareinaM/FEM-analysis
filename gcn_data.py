import os
import torch
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
from tqdm import tqdm

# find . -name '*.DS_Store' -type f -delete

data_path = {
    "train_y": "fem/train/y/",
    "train_adj": "fem/train/adj/",
    "train_x": "fem/train/x/",
    "test_y": "fem/test/y/",
    "test_adj": "fem/test/adj/",
    "test_x": "fem/test/x/"
}

def to_sparse(E):
    data = np.ones([len(E) * 2]) #np.zeros([len(E) * 2]) + 1
    row = E[:, 0]
    col = E[:, 1]
    row1 = np.append(row, col)
    col1 = np.append(col, row)
    adj1 = sp.coo_matrix((data, (row1, col1)))
    return adj1

def collect_data_train(path_adj, path_x, path_y):
    ret = []
    # list of str
    files_y = [f for f in os.listdir(path_y) if os.path.isfile(os.path.join(path_y, f))]
    files_adj = [f for f in os.listdir(path_adj) if os.path.isfile(os.path.join(path_adj, f))]
    files_x = [f for f in os.listdir(path_x) if os.path.isfile(os.path.join(path_x, f))]
    #print(len(files_x),len(files_y))

    num_files = len(files_x)
    print("number of files = ", num_files)

    for i in range(num_files):
        #print(filename)
        filename_x = files_x[i]
        filename_y = files_y[i]
        filename_adj = files_adj[i]

        file_path_y = path_y+filename_y
        file_path_x = path_x + filename_x
        file_path_adj = path_adj + filename_adj

        #print(filename_adj, filename_x, filename_y)
        y_f = open(file_path_y)
        x_f = open(file_path_x)
        adj_f = open(file_path_adj)

        with adj_f as file:
            e = np.array([[int(digit) for digit in line.split()] for line in file])
            adj_sp = to_sparse(e)
        with y_f as file:
            y = np.array([[float(digit) for digit in line.split()] for line in file])
        with x_f as file:
            x = np.array([[float(digit) for digit in line.split()] for line in file])

        """ 
        #files * 3 lists of [adj, x, y]
        """
        ret.append([adj_sp, x, y])
    return ret

class FEMDataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        adj, x, y = self.data[idx]
        adj = adj.toarray()
        adj[adj > 0] = 1
        return adj, torch.from_numpy(x), torch.from_numpy(y)

"""
# 38 * 3 lists of [adj, x, y]
ret = collect_data_train(data_path["train_adj"],data_path["train_x"],data_path["train_y"])

print(len(ret), len(ret[0]), ret[0][0].shape, len(ret[0][1]), len(ret[0][2]))

train_set = FEMDataset(ret)
train_data_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, num_workers=4)

for data in tqdm(train_data_loader, ascii=True):
#  [260, 260]
#     [260, 7]
#          [260, 4]
    adj, x, target = data[0][0], data[1][0], data[2][0]
    print("adj size = ",adj.size())
    print("x size = ", x.size())
    print("y size = ", target.size())
    #print(x[:10])
 """