import argparse
import os
import torch
import numpy as np
from gcn_model import Net
from gcn_utils import *
import gcn_data as data

#### python gcn_forward.py --checkpoint checkpoints/checkpoint.best 1\[1\,\ 0\,\ 0\]
# python gcn_forward.py --checkpoint checkpoints/fb+cp_p=2.2197 fem/test/

def collect_data_forward(path_x, path_adj):
    ret = []

    x_f = open(path_x)
    adj_f = open(path_adj)

    with adj_f as file:
        e = np.array([[int(digit) for digit in line.split()] for line in file])
        adj_sp = data.to_sparse(e)
        adj = adj_sp.toarray()
        adj[adj > 0] = 1
    with x_f as file:
        x = np.array([[float(digit) for digit in line.split()] for line in file])

    ret.append(x)
    ret.append(adj)
    return ret

parser = argparse.ArgumentParser(description="FEM network")
parser.add_argument("path_folder", type=str)
parser.add_argument("--checkpoint", metavar="path_to_checkpoint", type=str, required=True, help="the path to checkpoint")
opt = parser.parse_args()

# load model
checkpoint = torch.load(opt.checkpoint, map_location='cpu')
loss = checkpoint["psnr"]
loss = format(loss, '.4f')

# init model
model = Net()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

path_adj = opt.path_folder + "adj/"
path_x = opt.path_folder + "x/"

files_adj = [f for f in os.listdir(path_adj) if os.path.isfile(os.path.join(path_adj, f))]
files_x = [f for f in os.listdir(path_x) if os.path.isfile(os.path.join(path_x, f))]
num_files = len(files_x)

for i in range(num_files):
    filename_x = files_x[i]
    filename_adj = files_adj[i]

    file_path_x = path_x + filename_x
    file_path_adj = path_adj + filename_adj

    # load x
    open_file = lambda x, adj: collect_data_forward(x, adj)
    input = open_file(file_path_x, file_path_adj)

    # process input
    with torch.no_grad():
        output = model(torch.from_numpy(input[0]), torch.from_numpy(input[1]))

    output = output.numpy()
    # save result
    output_name = os.path.splitext(filename_x)
    tmp = output_name[0]
    tmp1 = tmp.split()
    fix = tmp1[1]
    force = str(tmp1[2]) + " " + str(tmp1[3]) + " " + str(tmp1[4])
    output_name = opt.path_folder+"y_" + str(loss) + " " + str(fix) + " " + force + ".txt"
    print(output_name)


    doc2 = open(output_name,'w')
    for item in output:
        for i in item:
            print(i, file=doc2, end = ' ')
        print("", file=doc2)
    doc2.close()

"""
filename_prev = "fem/test/"
filename_x = filename_prev + "x/cuboid 2 " + opt.filename + ".txt"
filename_adj = filename_prev + "adj/cuboid 2 " + opt.filename + ".txt"
"""