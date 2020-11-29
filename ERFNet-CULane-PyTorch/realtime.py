from PIL import Image,ImageDraw, ImageFilter, ImageColor, ImageOps
from erf_settings import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import models
#from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

# Alternative to matlab script that converts probability maps to lines
YS = IN_IMAGE_H - np.arange(POINTS_COUNT) * 20 - 1

save = False

def GetLane(score, thr = 0.3):

    coordinate = np.zeros(POINTS_COUNT)
    for i in range (POINTS_COUNT):
        lineId = int(TRAIN_IMG_H - i * 20 / IN_IMAGE_H_AFTER_CROP * TRAIN_IMG_H - 1)
        line = score[lineId, :]
        max_id = np.argmax(line)
        max_values = line[max_id]
        if max_values / 255.0 > thr:
            coordinate[i] = max_id

    coordSum = np.sum(coordinate > 0)
    if coordSum < 2:
        coordinate = np.zeros(POINTS_COUNT)

    return coordinate, coordSum


def GetLines(existArray, scoreMaps, thr = 0.3):
    coordinates = []

    for l in range(len(scoreMaps)):
        if (existArray[l]):
            coordinate, coordSum = GetLane(scoreMaps[l], thr)

            if (coordSum > 1):
                xs = coordinate * (IN_IMAGE_W / TRAIN_IMG_W)
                xs = np.round(xs).astype(np.int)
                pos = xs > 0
                curY = YS[pos]
                curX = xs[pos]
                curX += 1
                coordinates.append(list(zip(curX, curY)))
            else:
                coordinates.append([])
        else:
            coordinates.append([])

    return coordinates

def AddMask(img, mask, color, threshold = 0.3):
    back = Image.new('RGB', (img.size[0], img.size[1]), color=color)

    alpha = np.array(mask).astype(float) / 255
    alpha[alpha > threshold] = 1.0
    alpha[alpha <= threshold] = 0.0
    alpha *= 255
    alpha = alpha.astype(np.uint8)
    mask = Image.fromarray(np.array(alpha), 'L')
    mask_blur = mask.filter(ImageFilter.GaussianBlur(3))

    res = Image.composite(back, img, mask_blur)
    return res

def AddLinesPts(img, coords, color):
    base = ImageDraw.Draw(img)

    for i in range(0,4):
        if coords[i]:
            for pt in coords[i]:
                base.ellipse((pt[0], pt[1], pt[0]+10, pt[1]+10), fill=color, outline=(0,0,0))

    return 0

def run_forward(input_img, img_name, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()

    # switch to evaluate mode
    model.eval()

    input_var = torch.autograd.Variable(input_img)

    # compute output
    output, output_exist = model(input_var)

    # measure accuracy and record loss
    output = F.softmax(output, dim=1)

    pred = output.data.cpu().numpy()[0] # CxHxW
    pred_exist = output_exist.data.cpu().numpy()[0] # BxO
    exist = []
    prob_maps = []

    if save:
        directory = 'predicts/vgg_SCNN_DULR_w9' + img_name[:-10]
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_exist = open('predicts/vgg_SCNN_DULR_w9'+img_name.replace('.jpg', '.exist.txt'), 'w')
        for num in range(4):
            prob_map = (pred[num+1]*255).astype(int)
            save_img = cv2.blur(prob_map,(9,9))
            cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name.replace('.jpg', '_'+str(num+1)+'_avg.png'), save_img)
            if pred_exist[num] > 0.5:
                file_exist.write('1 ')
            else:
                file_exist.write('0 ')
        file_exist.close()
    else:
        for num in range(4):
            prob_map = (pred[num+1]*255).astype(int)
            prob_map = cv2.blur(prob_map,(9,9))
            prob_maps.append(prob_map.astype(np.uint8))
            if pred_exist[num] > 0.5:
                exist.append(1)
            else:
                exist.append(0)

    return prob_maps, exist

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

gpus = [0]
resume = "trained/ERFNet_trained.tar"
evaluate = False
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in gpus)
gpus = len(gpus)

num_class = 5
ignore_label = 255

model = models.ERFNet(num_class)
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
model = torch.nn.DataParallel(model, device_ids=range(gpus)).cuda()

if resume:
    if os.path.isfile(resume):
        print(("=> loading checkpoint '{}'".format(resume)))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_mIoU = checkpoint['best_mIoU']
        torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
        print(("=> loaded checkpoint '{}' (epoch {})".format(evaluate, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(resume)))

cudnn.benchmark = True
cudnn.fastest = True

# Experiment name
exp = 'vgg_SCNN_DULR_w9'
# Data root
data = './data/CULane'
# Directory where prob imgs generated by CNN are saved.
probRoot = './predicts/' + exp
# Directory to save fitted lanes.
output = './output/' + exp

testList = data + '/list/test.txt'

with open(testList) as file:
    test_fnames = [line.rstrip('\n') for line in file]
num = len(test_fnames)

plt.ion()
preproc_time = 0
model_time = 0
postproc_time = 0

for i in range(0,num):
    if i%100 == 99:
        print('Processing the {0} the image'.format(i+1))

    start_time = time.time()
    test_fname = test_fnames[i] # The test image
    test_img = cv2.imread(data + test_fname)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # run the neural net
    transform=torchvision.transforms.Compose([
            transforms.Normalize(input_mean, input_std)])
    input_img = test_img[VERTICAL_CROP_SIZE:, :, :]
    input_img = cv2.resize(input_img, (TRAIN_IMG_W, TRAIN_IMG_H), interpolation=cv2.INTER_LINEAR)
    input_img = torch.from_numpy(input_img).permute(2, 0, 1).contiguous().float()
    input_img.unsqueeze_(0)
    input_img = transform(input_img)

    preproc_time = time.time() - start_time
    start_time = time.time()

    scoreMaps, exist = run_forward(input_img, test_fname, model)

    model_time = time.time() - start_time
    start_time = time.time()

    if not save:
        # Get the coordinates of each lane that exists
        masked_img = Image.fromarray(test_img)           
        coordinates = GetLines(exist, scoreMaps)
        AddLinesPts(masked_img, coordinates, 'Green')

        postproc_time = time.time() - start_time
        start_time = time.time()

        plt.imshow(masked_img)
        plt.draw()
        #plt.pause(0.0333)
        plt.pause(0.001)
        plt.clf()
        print(preproc_time, model_time, postproc_time)
    total_time = preproc_time+model_time+postproc_time
    print(total_time, 1/total_time)
    