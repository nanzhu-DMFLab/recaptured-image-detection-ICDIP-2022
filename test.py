import numpy as np
from PIL import Image
import os
import torch
import glob 
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from model import MrcNet


# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# The path of data and log
data_root = './data/test_image/'
project_root = './log/'

patch_size = 96

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
        transforms.TenCrop(patch_size),   # # this is a list of PIL Images
        # use FiveCrop or TenCrop, the output is 5D (bs, ncrops, c, h, w) tensor,
        # it is necessary to change it into 4D(bs*ncrops, c, h, w)
        # returns a 4D tensor
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),  
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])


# instantiate model and initialize weights
model = MrcNet()
model.cuda()
checkpoint = torch.load(project_root + '/checkpoint_RIDICASSP_357_400.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

imageTmp = []
testTmp = []

testImageDir = data_root + 'NI'
testImageFile = list(glob.glob(testImageDir + '/*.png'))

len_NI=len(testImageFile)

testImageDir = data_root + 'RI'
testImageFile += list(glob.glob(testImageDir + '/*.png'))

for line in testImageFile:
    image_path = line
    lists = image_path.split('/')
    if lists[-2] == 'NI':
        testClass = 1
    else:
        testClass = 0

    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        test_input = transform(img)
        test_input = test_input.cuda()
        #input_var = Variable(test_input, volatile=True)
        with torch.no_grad():
            input_var = Variable(test_input)

        ncrops, c, h, w = input_var.size()   # bs, ncrops, c, h, w = input.size()
        # compute output
        output = model(input_var.view(-1, c, h, w))   # fuse batch size and ncrops
        # _, pred = torch.max(output, 1)
        pred = F.softmax(output, dim=1)
        mean = torch.mean(pred, dim=0)   # size: 1*2
        label = 0
        if mean[1] > 0.5:
            label = 1
        testTmp.append(label)  # the predicted label
        imageTmp.append(testClass)

imageLabelNp = np.array(imageTmp)
testLabelNp = np.array(testTmp)

#  Computing average accuracy on patches
result = imageLabelNp == testLabelNp

NI_result = result[:len_NI]
RI_result = result[len_NI:]


print('NI accuracy is:', NI_result.sum()*100.0/len(NI_result))
print('RI accuracy is:', RI_result.sum()*100.0/len(RI_result))
print('The average accuracy is:', (NI_result.sum()*100.0/len(NI_result) + RI_result.sum()*100.0/len(RI_result))/ 2)
