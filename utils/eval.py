import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from codes.model import HED_VGG16_v2
from codes.measure import accuracy
from codes.utils import AverageMeter, ProgressMeter, bce2d
import time
import cv2
import json

def make_new_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def load_model(model, best_model):

    if os.path.isfile(best_model):
        checkpoint = torch.load(best_model, map_location='cuda:0')
        state_dict = make_new_state_dict(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
        print("=> loaded best models from {} at epoch {:03d}".format(best_model, checkpoint['epoch']))

    return model


class PoseDatasetMultiTask(Dataset):
    def __init__(self, imdb_images, mode, pname, image_size):
        self.imdb_images  = imdb_images
        self.mode= mode
        self.pname = pname
        self.image_files = self._load_image_files()
        self.image_size = image_size

    def __getitem__(self, index):
        image_file = self.image_files[index]
        x_tensor, y1_tensor, y2_tensor = self._load_data(image_file)

        return  x_tensor, y1_tensor, y2_tensor

    def __len__(self):
        return len(self.image_files)

    def _load_image_files(self):
        image_files_path = os.path.join(self.imdb_images,  self.mode + '.txt')
        with open(image_files_path,'r') as f :
            image_files = f.read().splitlines()

        return image_files

    def _load_data(self, image_file):
        image_path = os.path.join(self.imdb_images, image_file)
        basename = os.path.basename(image_file).split('.')[0]
        annos_file = image_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.json')
        assert os.path.exists(annos_file), 'target file exists not'
        lsd_file  = image_path.replace('JPEGImages', 'JPEGImages_lsd')
        assert os.path.exists(lsd_file), 'lsd file exists not'
        im1 = cv2.imread(os.path.join(self.imdb_images, image_file))
        im1 = self._resize(im1, self.image_size)[:, :, ::-1] / 255.
        im2 = cv2.imread(lsd_file, cv2.IMREAD_GRAYSCALE)
        im2 = self._resize(im2, self.image_size)
        im2 = im2[:, :, np.newaxis]/255.
        im_tensor = torch.from_numpy(im1).permute(2, 0, 1).float()
        lsd_tensor = torch.from_numpy(im2).permute(2, 0, 1).float()
        with open(annos_file, 'r') as f:
            em_dict = json.load(f)
        em = (np.asarray(em_dict['euler angle']) + 90) / 180

        pose_tensor = torch.from_numpy(em).float().view(-1)

        return im_tensor, lsd_tensor, pose_tensor

    def _resize(self, img, scale_size):
        scale_factor = scale_size / max(img.shape)
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
        small_boundary = min(width, height)
        start = int((scale_size-small_boundary)/2)
        end = scale_size - start - small_boundary
        # resize image
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if width >= height:
            img_resized = cv2.copyMakeBorder(img_resized, start, end, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif height >= width:
            img_resized = cv2.copyMakeBorder(img_resized, 0, 0, start, end, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return img_resized


def eval_multitask(val_loader, model, criterion, measure, save_path):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':.4f')
    hed_losses = AverageMeter('HLoss', ':.4f')
    pose_losses = AverageMeter('PLoss', ':.4f')
    acc_pitch = AverageMeter('Pitch', ':.4f')
    acc_yaw = AverageMeter('Yaw', ':.4f')
    acc_roll = AverageMeter('Roll', ':.4f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, hed_losses, pose_losses, acc_pitch, acc_yaw, acc_roll],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    summary=[]
    with torch.no_grad():
        end = time.time()
        for i, (images, htarget, ptarget) in enumerate(val_loader):
            images = images.cuda()
            htarget = htarget.cuda()
            ptarget = ptarget.cuda()

            d1, d2, d3,  d6,  output = model(images)
            loss1 = bce2d(d1, htarget)
            loss2 = bce2d(d2, htarget)
            loss3 = bce2d(d3, htarget)
            loss6 = bce2d(d6, htarget)
            hed_loss = loss1 + loss2 + loss3 + loss6
            loss_pitch = criterion(output[:, 0], ptarget[:, 0])
            loss_yaw = criterion(output[:, 1], ptarget[:, 1])
            loss_roll = criterion(output[:, 2], ptarget[:, 2])
            pose_loss = criterion(output[:, 1], ptarget[:, 1])
            loss = hed_loss + pose_loss

            losses.update(loss.item(), images.size(0))
            hed_losses.update(hed_loss, images.size(0))
            pose_losses.update(pose_loss, images.size(0))
            acc_pitch.update(loss_pitch.item() * 180, images.size(0))
            acc_yaw.update(loss_yaw.item() * 180, images.size(0))
            acc_roll.update(loss_roll.item() * 180, images.size(0))
            summary.append([output.view(-1).cpu().numpy(), ptarget.view(-1).cpu().numpy()])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(i)

    return [losses.avg, hed_losses.avg, pose_losses.avg], [acc_pitch.avg, acc_yaw.avg, acc_roll.avg], summary


def resize(img, scale_size):
    scale_factor = scale_size / max(img.shape)
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)
    small_boundary = min(width, height)
    start = int((scale_size-small_boundary)/2)
    end = scale_size - start - small_boundary
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if width >= height:
        img_resized = cv2.copyMakeBorder(img_resized, start, end, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif height >= width:
        img_resized = cv2.copyMakeBorder(img_resized, 0, 0, start, end, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img_resized



def main():
    model_dir = './models/'
    imdb_images = './datasets/traindata/'
    mode = 'test'
    pname=''
    model = HED_VGG16_v2()
    test_dataset = PoseDatasetMultiTask(imdb_images=imdb_images, mode=mode, pname=pname, image_size=720)
    best_model=os.path.join(model_dir, 'checkpoint.pth.tar')
    model = load_model(model, best_model).cuda()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    loss, accs, summary = eval_multitask(test_loader, model, nn.L1Loss().cuda(), accuracy, save_path=model_dir)
    print(loss, accs)
    np.save(os.path.join(model_dir, 'predictions.npy'), np.asarray(summary))


if __name__ == '__main__':
    main()