import torch

class AverageMeter(object):
    """calculate and store the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """calculate the accuracy over the k top predictions"""
    res = []
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        pred = output.topk(maxk, 1, True, True)[1]
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_on_others(train_loader, model, criterion, optimizer, device):
    """train model on cifar10/100 or cub200 dataset"""
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

def validate_on_others(val_loader, model, criterion, device):
    """validate model on cifar10/100 or cub200 dataset"""
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

def train_on_imagenet(train_loader, model, criterion, optimizer, device, epoch, total_epochs, logger):
    """train model on imagenet dataset"""
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

def validate_on_imagenet(val_loader, model, criterion, device):
    """validate model on imagenet dataset"""
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