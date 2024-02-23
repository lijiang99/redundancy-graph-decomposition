import os
import argparse
import platform
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg19_bn
from utils.data import load_cub200
from utils.calculate import AverageMeter, accuracy
from datetime import datetime
import logging

parser = argparse.ArgumentParser(description="Train Model on CUB-200 from Scratch")

parser.add_argument("--arch", type=str, default="vgg19_bn", help="model architecture")
parser.add_argument("--dataset", type=str, default="cub200", help="dataset")
parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=32, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
parser.add_argument("--step-size", type=int, default=25, help="learning rate decay step size")

def train(train_loader, model, criterion, optimizer, device):
    losses = AverageMeter("loss")
    top1 = AverageMeter("acc@1")
    
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        logits = model(images)
        loss = criterion(logits, target)
        prec1 = accuracy(logits, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses.avg, top1.avg

def validate(val_loader, model, criterion, device):
    losses = AverageMeter("loss")
    top1 = AverageMeter("acc@1")
    
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            loss = criterion(logits, target)
            prec1 = accuracy(logits, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
    
    return losses.avg, top1.avg

def main():
    args = parser.parse_args()
    
    # set for log file
    log_dir = os.path.join(args.dataset, "log", "pre-train")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{args.arch}.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    logger.info(f"author: jiang li - task: train {args.arch} on {args.dataset} from scratch")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => printing arguments settings")
    args_info = str(args).replace(" ", "\n  ").replace("(", "(\n  ").replace(")", "\n)")
    logger.info(f"{args_info}")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => printing running environment")
    logger.info(f"{'python':<6} version: {platform.python_version()}")
    logger.info(f"{'torch':<6} version: {torch.__version__}")
    logger.info(f"{'cuda':<6} version: {torch.version.cuda}")
    logger.info(f"{'cudnn':<6} version: {torch.backends.cudnn.version()}")
    device = torch.device("cuda")
    device_prop = torch.cuda.get_device_properties(device)
    logger.info(f"{'device':<6} version: {device_prop.name} ({device_prop.total_memory/(1024**3):.2f} GB)")
    
    # load dataset and create model
    num_classes = 200
    dataset_dir = os.path.join(args.dataset, "dataset")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading dataset from '{dataset_dir}'")
    train_loader, val_loader = eval("load_"+args.dataset)(dataset_dir, batch_size=args.batch_size)
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => creating model '{args.arch}'")
    model = eval(args.arch)(pretrained=True).to(device)
    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes).to(device)
    logger.info(str(model))
    
    # set hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set the save path of the model
    pretrain_dir = os.path.join(args.dataset, "pre-train")
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)
    save_path = os.path.join(pretrain_dir, f"{args.arch}-weights.pth")
    
    # start training
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => training model '{args.arch}'")
    best_acc = 0
    for epoch in range(args.epochs):
        beg_time = datetime.now()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        end_time = datetime.now()
        lr = optimizer.param_groups[0]["lr"]
        consume_time = int((end_time-beg_time).total_seconds())
        train_message = f"Epoch[{epoch+1:0>2}/{args.epochs}] - time: {consume_time}s - lr: {lr} - loss: {train_loss:.2f} - prec@1: {train_acc:.2f}"
        logger.info(train_message)
        valid_loss, valid_acc = validate(val_loader, model, criterion, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), save_path)
        logger.info(f"Test - acc@1: {valid_acc:.2f} - best acc: {best_acc:.2f}")
        scheduler.step()
    
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => done!")

if __name__ == "__main__":
    main()
