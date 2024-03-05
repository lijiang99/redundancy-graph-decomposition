import os
import argparse
import torch
import torch.nn as nn
import torchvision
from large_scale.models import vgg16_bn, vgg19_bn, resnet50
from large_scale.pruning import prune_vggnet_weights, prune_resnet_weights
from utils.data import load_imagenet
from utils.calculate import train_on_imagenet, validate_on_imagenet, evaluate
from utils.logger import Logger
import json

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

def main():
    args = parser.parse_args()
    pruned_model_str = f"{args.arch}-{args.threshold}"
    
    # set for log file
    log_dir = os.path.join(args.root, args.dataset, "log", "fine-tune")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{pruned_model_str}.log")
    logger = Logger(log_path)
    
    logger.task(f"fine-tune pruned {args.arch} (threshold={args.threshold}) on {args.dataset}")
    logger.hint("printing arguments settings")
    logger.args(args)
    logger.hint("printing running environment")
    device = torch.device("cuda")
    logger.envs(device)
    
    # load pre-trained weights and model
    origin_model = eval(f"torchvision.models.{args.arch}")(pretrained=True).to(device)
    origin_state_dict = origin_model.state_dict()
    dataset_dir = os.path.join(args.root, args.dataset, "dataset")
    logger.hint(f"loading dataset from '{dataset_dir}'")
    train_loader, val_loader = load_imagenet(dataset_dir, batch_size=args.batch_size)
    
    # load pruning information
    prune_info_path = os.path.join(args.root, args.dataset, "prune-info", f"{pruned_model_str}.json")
    logger.hint(f"loading pruning information from '{prune_info_path}'")
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
    logger.hint(f"creating pruned model '{pruned_model_str}'")
    pruned_model = eval(args.arch)(num_classes=1000, mask_nums=[value["mask_num"] for value in prune_info.values()]).to(device)
    pruned_state_dict = pruned_model.state_dict()
    logger.mesg(str(pruned_model))
    
    # load pruned weights to pruned model
    logger.hint(f"loading pruned weights to pruned model '{pruned_model_str}'")
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
    logger.hint(f"fine-tuning pruned model '{pruned_model_str}'")
    best_top1_acc, best_top5_acc = 0, 0
    for epoch in range(args.epochs):
        train_loss, train_top1_acc, train_top5_acc = train_on_imagenet(train_loader, pruned_model, criterion, optimizer, device, epoch, args.epochs, logger)
        valid_loss, valid_top1_acc, valid_top5_acc = validate_on_imagenet(val_loader, pruned_model, criterion, device)
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            torch.save(pruned_model.state_dict(), save_path)
        logger.mesg(f"Test - top@1: {valid_top1_acc:.2f} - top@5: {valid_top5_acc:.2f} - best accuracy (top@1, top@5): ({best_top1_acc:>5.2f}, {best_top5_acc:>5.2f})")
        scheduler.step()
    
    # evaluate pruning effect
    logger.hint(f"evaluating pruned model '{pruned_model_str}'")    
    origin_result = evaluate(origin_model, 224, validate_on_imagenet, val_loader, criterion, device)
    pruned_result = evaluate(pruned_model, 224, validate_on_imagenet, val_loader, criterion, device)
    logger.eval(origin_result, pruned_result)
    logger.hint("done!")

if __name__ == "__main__":
    main()
