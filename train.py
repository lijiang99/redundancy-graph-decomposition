import os
import argparse
import torch
import torch.nn as nn
import torchvision
import small_scale.models
from utils.data import load_cifar10, load_cifar100, load_cub200
from utils.calculate import train_on_others, validate_on_others
from utils.logger import Logger
from datetime import datetime
import math

parser = argparse.ArgumentParser(description="Train Model on CIFAR-10/100 or CUB-200 from Scratch")

parser.add_argument("--root", type=str, default="./", help="project root directory")
parser.add_argument("--arch", type=str, default="vgg16_bn", help="model architecture")
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--step-size", type=int, default=50, help="learning rate decay step size")

def main():
    args = parser.parse_args()
    
    # set for log file
    log_dir = os.path.join(args.root, args.dataset, "log", "pre-train")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{args.arch}.log")
    logger = Logger(log_path)
    
    logger.task(f"train {args.arch} on {args.dataset} from scratch")
    logger.hint("printing arguments settings")
    logger.args(args)
    logger.hint("printing running environment")
    device = torch.device("cuda")
    logger.envs(device)
    
    # load dataset and create model
    dataset_dir = os.path.join(args.root, args.dataset, "dataset")
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    logger.hint(f"loading dataset from '{dataset_dir}'")
    train_loader, val_loader = eval(f"load_{args.dataset}")(dataset_dir, batch_size=args.batch_size)
    logger.hint(f"creating model '{args.arch}'")
    model = None
    if args.dataset == "cub200":
        model = eval(f"torchvision.models.{args.arch}")(pretrained=True).to(device)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=200).to(device)
    else:
        model = eval(f"small_scale.models.{args.arch}")(num_classes=(10 if args.dataset == "cifar10" else 100)).to(device)
    logger.mesg(str(model))
    
    # set hyperparameters
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set the save path of the model
    pretrain_dir = os.path.join(args.root, args.dataset, "pre-train")
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)
    save_path = os.path.join(pretrain_dir, f"{args.arch}-weights.pth")
    
    # start training
    logger.hint(f"training model '{args.arch}'")
    best_top1_acc, ndigits, width = 0, int(abs(math.log10(args.learning_rate))), int(math.log10(args.epochs)+1)
    for epoch in range(args.epochs):
        beg_time = datetime.now()
        train_loss, train_top1_acc = train_on_others(train_loader, model, criterion, optimizer, device)
        end_time = datetime.now()
        lr = round(optimizer.param_groups[0]["lr"], epoch//args.step_size+ndigits)
        consume_time = int((end_time-beg_time).total_seconds())
        train_message = f"Epoch[{epoch+1:0>{width}}/{args.epochs}] - time: {consume_time:0>2}s - lr: {lr} - loss: {train_loss:.2f} - prec@1: {train_top1_acc:.2f}"
        logger.mesg(train_message)
        valid_loss, valid_top1_acc = validate_on_others(val_loader, model, criterion, device)
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            torch.save(model.state_dict(), save_path)
        logger.mesg(f"Test - top@1: {valid_top1_acc:.2f} - best accuracy (top@1): {best_top1_acc:.2f}")
        scheduler.step()
    
    logger.hint("done!")

if __name__ == "__main__":
    main()
