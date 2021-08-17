import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from gcn_model import Net
import copy
from gcn_data import *
from gcn_utils import *

# python gcn_debug.py train --num-epoch=1 --save-freq=1
# python gcn_debug.py resume checkpoints/checkpoint.best --num-epoch=10
# python gcn_debug.py inspect checkpoints/debug_checkpoint.best

torch.autograd.set_detect_anomaly(True)
num_workers = 4

parser = ArgParser(description="GCN")
subparsers = parser.add_subparsers(dest="mode", required=True, help="sub commands")

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

record = {
    "cur": 0,
    "best": 100,
    "epoch": 0,
    "model": None,
    "optim": None
}

start_epoch = 0
checkpoint = None
if opt.mode == "resume":
    checkpoint = torch.load(opt.checkpoint)
    print("Using checkpoint '{}'".format(opt.checkpoint))
    save_opt = checkpoint["opt"]
    start_epoch = checkpoint["epoch"] + 1
    for arg in vars(opt):
        val = getattr(opt, arg)
        if val == None:
            setattr(opt, arg, getattr(save_opt, arg))
elif opt.mode == "inspect":
    checkpoint = torch.load(opt.checkpoint)
    print("Saved at epoch {} with PSNR={:.4f}".format(checkpoint["epoch"], checkpoint["psnr"]))
    print("Trained with arguments:", checkpoint["opt"])
    exit(0)


model = Net()

if checkpoint:
    checkpoint_ = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint_["model_state_dict"])

# optimizer
# params: iterable of parameters to optimize or dicts defining parameter groups
optimizer = optim.Adagrad(model.parameters())
if checkpoint:
    optimizer.load_state_dict(checkpoint_["optimizer_state_dict"])

# data
ret = collect_data_train("fem/debug/adj/", "fem/debug/x/", "fem/debug/y/")

train_set = FEMDataset(ret)
test_set =FEMDataset(ret)

train_data_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, num_workers=num_workers)
test_data_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, num_workers=num_workers)

# loss func
loss_function = nn.L1Loss()

def train(epoch, data_loader):
    # Sets the module in training mode
    model.train()
    loss_metric = AvgMetric(len(data_loader))
    for data in tqdm(data_loader, ascii=True):
        optimizer.zero_grad()

        adj, x, target = data[0][0], data[1][0], data[2][0]
        output = model(x, adj)

        loss = loss_function(output, target)

        #print(loss)
        loss.backward()
        optimizer.step()

        loss_metric.add(loss.item())

    avg_loss = loss_metric.average()
    avg_loss = format(avg_loss, '.4f')
    print("[Train] Epoch {cur}/{total} complete: Avg. Loss={val:.4f}".format(cur=epoch, total=opt.num_epoch-1, val=loss_metric.average()))
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

            psnr = 10 * mae #10 * torch.log10(mae)
            #print("mae = ",mae, "  psnr = ",psnr)
            psnr_metric.add(psnr.item())

        record["cur"] = psnr_metric.average()

        if record["cur"] < record["best"]:
            record["best"] = record["cur"]
            record["epoch"] = epoch
            record["model"] = copy.deepcopy(model.state_dict())
            record["optim"] = copy.deepcopy(optimizer.state_dict())

        print("[Test] Epoch {cur}/{total} complete: Avg. PSNR={val:.4f}".format(cur=epoch, total=opt.num_epoch-1, val=psnr_metric.average()))

def save_checkpoint1(epoch, model, optimizer, psnr=0, opt=None, path=None, best=False):
    psnr1 = format(psnr, '.4f')
    if path is None:
        path = "checkpoints/debug_checkpoint.best" if best else "checkpoints/debug_checkpoint_p={psnr}".format(psnr=psnr1)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model if best else model.state_dict(),
            "optimizer_state_dict": optimizer if best else optimizer.state_dict(),
            "opt": opt,
            "psnr": psnr
        },
        path,
    )

best_mae = 1.0

# train num_epoch-start_epoch+1 epochs
for epoch in range(start_epoch, opt.num_epoch):
    train(epoch, train_data_loader)
    #mae_avg = float(mae_avg)
    test(epoch, test_data_loader)
    if (epoch + 1) % opt.save_freq == 0: # and mae_avg < best_mae:
        #best_mae = mae_avg
        # save a checkpoint
        save_checkpoint1(epoch, model, optimizer, opt=opt, psnr=record["cur"])

if record["model"] != None:
    # save best checkpoint
    print("Best PSNR={:.4f}".format(record["best"]))
    save_checkpoint1(record["epoch"], record["model"], record["optim"], psnr=record["best"], best=True, opt=opt)





