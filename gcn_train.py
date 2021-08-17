import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from gcn_model import Net
import copy
import sys
#import warnings
from gcn_data import *
from gcn_utils import *

# python gcn_train.py train --num-epoch=20 --save-freq=2
# python gcn_train.py resume checkpoints/checkpoint.best --num-epoch=10 --save-freq=2
# python gcn_train.py inspect checkpoints/checkpoint.best

"""
def supress_warning():
    # suppress user warning
    warnings.simplefilter("ignore", UserWarning)
supress_warning()
"""
torch.autograd.set_detect_anomaly(True)
num_workers = 4

parser = ArgParser(description="GCN")
subparsers = parser.add_subparsers(dest="mode", help="sub commands")

train_parser = subparsers.add_parser("train", help="train SRCNN model")
resume_parser = subparsers.add_parser("resume", help="resume training")
inspect_parser = subparsers.add_parser("inspect", help="inspect a checkpoint")

train_parser.add_argument("--num-epoch", type=int, default=400, help="number of epochs to train for")
train_parser.add_argument("--save-freq", type=int, default=20, help="save checkpoint every SAVE_FREQ epoches")
resume_parser.add_argument("--num-epoch", type=int, help="number of epochs to train for")
resume_parser.add_argument("--save-freq", type=int, help="save checkpoint every SAVE_FREQ epoches")
resume_parser.add_argument("checkpoint", type=str, help="path to checkpoint")
inspect_parser.add_argument("checkpoint", type=str, help="path to checkpoint")
opt = parser.parse_args()

start_epoch = 0
checkpoint = None
if opt.mode == "resume":
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    print("Using checkpoint '{}'".format(opt.checkpoint))
    save_opt = checkpoint["opt"]
    start_epoch = checkpoint["epoch"] + 1
    for arg in vars(opt):
        val = getattr(opt, arg)
        if val == None:
            setattr(opt, arg, getattr(save_opt, arg))
elif opt.mode == "inspect":
    checkpoint = torch.load(opt.checkpoint, map_location='cpu')
    print("\nSaved at epoch {} with PSNR={:.4f}\n".format(checkpoint["epoch"], checkpoint["psnr"]))
    print("Trained with arguments:", checkpoint["opt"])
    sys.exit(0)

model = Net()

if checkpoint:
    checkpoint_ = torch.load(opt.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint_["model_state_dict"])

# optimizer
# params: iterable of parameters to optimize or dicts defining parameter groups
optimizer = optim.Adagrad(model.parameters())

if checkpoint:
    optimizer.load_state_dict(checkpoint_["optimizer_state_dict"])

# data
ret = collect_data_train(data_path["train_adj"], data_path["train_x"], data_path["train_y"])
#print("len = ",len(ret))
ret_test = collect_data_train(data_path["test_adj"], data_path["test_x"], data_path["test_y"])

train_set = FEMDataset(ret)
test_set =FEMDataset(ret_test)

train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=num_workers)
test_data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False, num_workers=num_workers)

# loss func
loss_function = nn.L1Loss()

record = {
    "cur": 0,
    "best": 100,
    "epoch": 0,
    "model": None,
    "optim": None
}

def train(epoch, data_loader):
    # Sets the module in training mode
    model.train()
    loss_metric = AvgMetric(len(data_loader))
    for data in data_loader:
        optimizer.zero_grad()

        adj, x, target = data[0][0], data[1][0], data[2][0]
        output = model(x, adj)

        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

        loss_metric.add(loss.item())
    avg_loss = loss_metric.average()
    avg_loss = format(avg_loss, '.4f')
    print("[Train] Epoch {cur}/{total} complete: Avg. Loss={val}".format(cur=epoch, total=opt.num_epoch-1, val=avg_loss))
    return avg_loss

# test procedure
def test(epoch, data_loader):
    model.eval()
    psnr_metric = AvgMetric(len(data_loader))
    with torch.no_grad():
        for data in data_loader:
            adj, x, target = data[0][0], data[1][0], data[2][0]
            output = model(x, adj)

            mae = loss_function(output, target)
            psnr = 10 * mae #10 * torch.log10(1 / mae)
            psnr_metric.add(psnr.item())

        record["cur"] = psnr_metric.average()

        if record["cur"] < record["best"]:
            record["best"] = record["cur"]
            record["epoch"] = epoch
            record["model"] = copy.deepcopy(model.state_dict())
            record["optim"] = copy.deepcopy(optimizer.state_dict())

        print("[Test] Epoch {cur}/{total} complete: Avg. PSNR={val:.4f}".format(cur=epoch, total=opt.num_epoch-1, val=psnr_metric.average()))

# train num_epoch-start_epoch+1 epochs
for epoch in range(start_epoch, opt.num_epoch):
    mae_avg = train(epoch, train_data_loader)
    test(epoch, test_data_loader)
    if (epoch + 1) % opt.save_freq == 0:
        # save a checkpoint
        save_checkpoint(epoch, model, optimizer, opt=opt, psnr=record["cur"])

if record["model"] != None:
    # save best checkpoint
    print("Best PSNR={:.4f}".format(record["best"]))
    save_checkpoint(record["epoch"], record["model"], record["optim"], psnr=record["best"], best=True, opt=opt)






