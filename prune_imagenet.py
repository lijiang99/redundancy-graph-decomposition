import os
import argparse
import platform
import torch
import torch.nn as nn
from large_scale.models import vgg16_bn, vgg19_bn, resnet50
from large_scale.pruning import prune_vggnet_weights, prune_resnet_weights
from utils.data import load_imagenet
from utils.calculate import AverageMeter, accuracy
from thop import profile
from datetime import datetime
import json
import logging

parser = argparse.ArgumentParser(description="Fine-tune Pruned Model on ImageNet")

parser.add_argument("--root", type=str, default="./", help="project root directory")
parser.add_argument("--arch", type=str, default="resnet50", help="model architecture")
parser.add_argument("--dataset", type=str, default="imagenet", help="dataset")
parser.add_argument("--threshold", type=float, default=0.7, help="similarity threshold")
parser.add_argument("--epochs", type=int, default=50, help="number of fine-tuning epochs")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
parser.add_argument("--step-size", type=int, default=20, help="learning rate decay step size")

def train(train_loader, model, criterion, optimizer, device, epoch, total_epochs, logger):
    losses = AverageMeter("loss")
    top1 = AverageMeter("acc@1")
    top5 = AverageMeter("acc@5")
    lr = optimizer.param_groups[0]["lr"]
    
    model.train()
    beg_time = datetime.now()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % (len(train_loader)//5) == 0:
            consume_time = str(datetime.now()-beg_time).split('.')[0]
            train_message = f"Epoch[{epoch+1:0>2}/{total_epochs}][{i+1}/{len(train_loader)}] - time: {consume_time} - lr: {lr} - loss: {losses.avg:.2f} - prec@1: {top1.avg:>5.2f} - prec@5: {top5.avg:>5.2f}"
            logger.info(train_message)
            beg_time = datetime.now()
    
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device):
    losses = AverageMeter("loss")
    top1 = AverageMeter("acc@1")
    top5 = AverageMeter("acc@5")
    
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
    
    return losses.avg, top1.avg, top5.avg

def main():
    args = parser.parse_args()
    pruned_model_str = f"{args.arch}-{args.threshold}"
    
    # set for log file
    log_dir = os.path.join(args.root, args.dataset, "log", "fine-tune")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{pruned_model_str}.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    logger.info(f"author: jiang li - task: fine-tune pruned {args.arch} (threshold={args.threshold}) on imagenet")
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
    num_classes = 1000
    pretrain_weights_path = os.path.join(args.root, args.dataset, "pre-train", f"{args.arch}-weights.pth")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading weights from '{pretrain_weights_path}'")
    origin_model = eval(args.arch)(num_classes=num_classes).to(device)
    origin_state_dict = torch.load(pretrain_weights_path, map_location=device)
    origin_model.load_state_dict(origin_state_dict)
    dataset_dir = os.path.join(args.root, args.dataset, "dataset")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading dataset from '{dataset_dir}'")
    train_loader, val_loader = load_imagenet(dataset_dir, batch_size=args.batch_size)
    
    # load pruning information
    prune_info_path = os.path.join(args.root, args.dataset, "prune-info", f"{pruned_model_str}.json")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruning information from '{prune_info_path}'")
    prune_info = None
    with open(prune_info_path, "r") as f:
        prune_info = json.load(f)
    
    # get name of conv layers and bn layers
    conv_layers, bn_layers, linear_layers = [], [], []
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
        elif isinstance(module, nn.BatchNorm2d):
            bn_layers.append(name)
        elif isinstance(module, nn.Linear):
            linear_layers.append(name)
    
    # create pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => creating pruned model '{pruned_model_str}'")
    pruned_model = eval(args.arch)(num_classes=num_classes, mask_nums=[value["mask_num"] for value in prune_info.values()]).to(device)
    pruned_state_dict = pruned_model.state_dict()
    logger.info(str(pruned_model))
    
    # load pruned weights to pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruned weights to pruned model '{pruned_model_str}'")
    if "vgg" in args.arch:
        pruned_state_dict = prune_vggnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "resnet" in args.arch:
        pruned_state_dict = prune_resnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    
    pruned_model.load_state_dict(pruned_state_dict)
    
    # set hyperparameters
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set the save path of the pruned model
    fine_tune_dir = os.path.join(args.root, args.dataset, "fine-tune")
    if not os.path.isdir(fine_tune_dir):
        os.makedirs(fine_tune_dir)
    save_path = os.path.join(fine_tune_dir, f"{pruned_model_str}-weights.pth")
    
    # start fine-tuning
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => fine-tuning pruned model '{pruned_model_str}'")
    best_top1_acc, best_top5_acc = 0, 0
    for epoch in range(args.epochs):
        train_loss, train_top1_acc, train_top5_acc = train(train_loader, pruned_model, criterion, optimizer, device, epoch, args.epochs, logger)
        valid_loss, valid_top1_acc, valid_top5_acc = validate(val_loader, pruned_model, criterion, device)
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            torch.save(pruned_model.state_dict(), save_path)
        logger.info(f"Test - acc@1: {valid_top1_acc:.2f} - acc@5: {valid_top5_acc:.2f} - best accuracy (top@1, top@5): ({best_top1_acc:>5.2f}, {best_top5_acc:>5.2f})")
        scheduler.step()
    
    # evaluate pruning effect
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => evaluating pruned model '{pruned_model_str}'")
    origin_top1_acc, origin_top5_acc = tuple(validate(val_loader, origin_model, criterion, device)[1:])
    pruned_top1_acc, pruned_top5_acc = best_top1_acc, best_top5_acc
    input_image_size = 224
    input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    origin_flops, origin_params = profile(origin_model, inputs=(input_image,))
    pruned_flops, pruned_params = profile(pruned_model, inputs=(input_image,))
    logger.info(f"{'top@1':<6}: {origin_top1_acc:>6.2f}% -> {pruned_top1_acc:>6.2f}% - drop: {origin_top1_acc-pruned_top1_acc:>5.2f}%")
    logger.info(f"{'top@5':<6}: {origin_top5_acc:>6.2f}% -> {pruned_top5_acc:>6.2f}% - drop: {origin_top5_acc-pruned_top5_acc:>5.2f}%")
    logger.info(f"{'flops':<6}: {origin_flops/1e9:>6.2f}G -> {pruned_flops/1e9:>6.2f}G - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
    logger.info(f"{'params':<6}: {origin_params/1e6:>6.2f}M -> {pruned_params/1e6:>6.2f}M - drop: {(origin_params-pruned_params)/origin_params*100:>5.2f}%")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => done!")

if __name__ == "__main__":
    main()