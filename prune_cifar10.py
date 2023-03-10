import os
import argparse
import platform
import torch
import torch.nn as nn
from cifar10.models import vgg16, densenet40, googlenet
from cifar10.models import resnet20, resnet32, resnet44, resnet56, resnet110
from cifar10.data import load_cifar10
from cifar10.pruning import prune_vggnet_weights
from cifar10.pruning import prune_resnet_weights
from cifar10.pruning import prune_densenet_weights
from cifar10.pruning import prune_googlenet_weights
from utils.calculate import AverageMeter, accuracy
from thop import profile
from datetime import datetime
import json
import logging

parser = argparse.ArgumentParser(description="Fine-tune Pruned Model on CIFAR-10")

parser.add_argument("--arch", type=str, default="vgg16", help="model architecture")
parser.add_argument("--pretrain-dir", type=str, default="./cifar10/pre-train/", help="pre-trained model saved directory")
parser.add_argument("--dataset-dir", type=str, default="./cifar10/dataset/", help="dataset saved directory")
parser.add_argument("--pruneinfo-dir", type=str, default="./cifar10/prune-info/", help="pruning information saved directory")
parser.add_argument("--saved-dir", type=str, default="./cifar10/fine-tune/", help="pruned model saved directory")
parser.add_argument("--log-dir", type=str, default="./cifar10/log/fine-tune/", help="log file saved directory")
parser.add_argument("--threshold", type=float, default=0.7, help="similarity threshold")
parser.add_argument("--epochs", type=int, default=90, help="number of fine-tuning epochs")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--adjust", type=int, default=30, help="learning rate decay step")

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
    pruned_model_str = f"{args.arch}-{args.threshold}"
    
    # set for log file
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    log_path = os.path.join(args.log_dir, f"{pruned_model_str}.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    logger.info(f"author: jiang li - task: fine-tune pruned {args.arch} (threshold={args.threshold}) on cifar10")
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
    
    # load pre-trained weights and model
    pretrain_weights_path = os.path.join(args.pretrain_dir, f"{args.arch}-weights.pth")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading weights from '{pretrain_weights_path}'")
    origin_model = eval(args.arch)().to(device)
    origin_state_dict = torch.load(pretrain_weights_path, map_location=device)
    origin_model.load_state_dict(origin_state_dict)
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading dataset from '{args.dataset_dir}'")
    train_loader, val_loader = load_cifar10(args.dataset_dir, batch_size=args.batch_size)
    
    # load pruning information
    prune_info_path = os.path.join(args.pruneinfo_dir, f"{pruned_model_str}.json")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruning information from '{prune_info_path}'")
    prune_info = None
    with open(prune_info_path, "r") as f:
        prune_info = json.load(f)
    
    # get name of conv layers and bn layers
    conv_layers, bn_layers = [], []
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append(name)
    
    # create pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => creating pruned model '{pruned_model_str}'")
    pruned_model = eval(args.arch)(mask_nums=[value["mask_num"] for value in prune_info.values()]).to(device)
    pruned_state_dict = pruned_model.state_dict()
    logger.info(str(pruned_model))
    
    # load pruned weights to pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruned weights to pruned model '{pruned_model_str}'")
    if "vgg" in args.arch:
        pruned_state_dict = prune_vggnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict,
                                                 origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers)
    elif "resnet" in args.arch:
        pruned_state_dict = prune_resnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict,
                                                 origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers)
    elif "densenet" in args.arch:
        pruned_state_dict = prune_densenet_weights(prune_info=prune_info,
                                                   pruned_state_dict=pruned_state_dict,
                                                   origin_state_dict=origin_state_dict,
                                                   conv_layers=conv_layers, bn_layers=bn_layers)
    elif "googlenet" in args.arch:
        pruned_state_dict = prune_googlenet_weights(prune_info=prune_info,
                                                    pruned_state_dict=pruned_state_dict,
                                                    origin_state_dict=origin_state_dict,
                                                    conv_layers=conv_layers, bn_layers=bn_layers)
    
    pruned_model.load_state_dict(pruned_state_dict)
    
    # set hyperparameters
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    decay_step = list(range(0, args.epochs, args.adjust)[1:])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set the save path of the pruned model
    if not os.path.isdir(args.saved_dir):
        os.makedirs(args.saved_dir)
    save_path = os.path.join(args.saved_dir, f"{pruned_model_str}-weights.pth")
    
    # start fine-tuning
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => fine-tuning pruned model '{pruned_model_str}'")
    best_acc = 0
    for epoch in range(args.epochs):
        beg_time = datetime.now()
        train_loss, train_acc = train(train_loader, pruned_model, criterion, optimizer, device)
        end_time = datetime.now()
        lr = optimizer.param_groups[0]["lr"]
        consume_time = int((end_time-beg_time).total_seconds())
        train_message = f"Epoch[{epoch+1:0>2}/{args.epochs}] - time: {consume_time}s - lr: {lr} - loss: {train_loss:.2f} - prec@1: {train_acc:.2f}"
        logger.info(train_message)
        valid_loss, valid_acc = validate(val_loader, pruned_model, criterion, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(pruned_model.state_dict(), save_path)
        logger.info(f"Test - acc@1: {valid_acc:.2f} - best acc: {best_acc:.2f}")
        scheduler.step()
    
    # evaluate pruning effect
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => evaluating pruned model '{pruned_model_str}'")
    origin_best_acc, pruned_best_acc = validate(val_loader, origin_model, criterion, device)[1], best_acc
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    origin_flops, origin_params = profile(origin_model, inputs=(input_image,))
    pruned_flops, pruned_params = profile(pruned_model, inputs=(input_image,))
    logger.info(f"{'acc':<6}: {origin_best_acc:>6.2f}% -> {pruned_best_acc:>6.2f}% - drop: {origin_best_acc-pruned_best_acc:>5.2f}%")
    if args.arch == "googlenet":
        logger.info(f"{'flops':<6}: {origin_flops/1e9:>6.2f}G -> {pruned_flops/1e9:>6.2f}G - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
    else:
        logger.info(f"{'flops':<6}: {origin_flops/1e6:>6.2f}M -> {pruned_flops/1e6:>6.2f}M - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
    logger.info(f"{'params':<6}: {origin_params/1e6:>6.2f}M -> {pruned_params/1e6:>6.2f}M - drop: {(origin_params-pruned_params)/origin_params*100:>5.2f}%")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => done!")

if __name__ == "__main__":
    main()