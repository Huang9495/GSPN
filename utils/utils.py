import torch
import torch.nn.functional as F
import shutil
import os
import time


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch in args.lr_step))
    args.lr = lr
    print(lr)
    # lr = args.lr * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def save_checkpoint(state, is_best, file_path):
    # timestr = time.strftime("%Y%m%d%H%M%S")
    filename = os.path.join(file_path, 'checkpoint.pth.tar')
    torch.save(state, os.path.join(file_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(filename, os.path.join(file_path, 'model_best.pth.tar'))


def make_new_state_dict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # remove `module.`

        name = k[7:]

        new_state_dict[name] = v

    return new_state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def bce2d(input, target):
    ## input, target: shape Bs x C x W x H
    n, c, h, w = input.size()

    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss