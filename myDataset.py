import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import torch
import itertools
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Note that, the sort() can change the label index
# using sort(): {'recaptured': 0, 'nature': 1}
# using sort(key=len): {'RI': 0, 'NI': 1}锛宬ey=len means according to the length of characters
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]   # output: classes=['CG', 'PG']
    classes.sort(key=len)
    # output {'RI': 0, 'NI': 1}
    class_to_idx = {classes[i]: i for i in range(len(classes))}   
    #print(classes)    # ['RI', 'NI']
    #print(class_to_idx)   # {'RI': 0, 'NI': 1}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    num_in_class = []  # the number of samples in each class
    images_txt = []
    dir = os.path.expanduser(dir)   # change '~ 'and '~user' into 'HOME/your_name'
    for target in sorted(os.listdir(dir), key=len):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            num = 0
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)   # output: ('XXX.png',1)
                    images_txt.append(target + '/' + fname)   # such as RI/xxx.png
                    num += 1
            num_in_class.append(num)

    return images, num_in_class, images_txt


def pil_loader(path, mode='RGB'):   # # open path as file to avoid ResourceWarning
    with open(path, 'rb') as f:  
        img = Image.open(f)
        if mode == 'L':
            return img.convert('L')  # convert image to grey
        elif mode == 'RGB':
            return img.convert('RGB')  # convert image to rgb image
        elif mode == 'HSV':
            return img.convert('HSV')
            # elif mode == 'LAB':
            #     return RGB2Lab(img)


def accimage_loader(path):
    # accimage is a partial replacement for PIL.Image (only RGB JPEG images, 
    # and only a subset of image transformations used in torch.vision) on top of JPEG-Turbo and Intel IPP. 
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':   
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


class MyDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):

        classes, class_to_idx = find_classes(root)
        imgs, num_in_class, images_txt = make_dataset(root, class_to_idx)
        
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                     "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        # self.mode = args.img_mode
        # self.input_nc = args.input_nc
        self.imgs = imgs
        self.num_in_class = num_in_class
        self.images_txt = images_txt
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, 'RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class RandomBalancedSampler(Sampler):
    """
    Samples elements randomly, with an arbitrary size, independant from dataset length.
    This is a balanced sampling that will sample the whole dataset with a random permutation.
    """
    def __init__(self, data_source):
        print('Using RandomBalancedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = torch.randperm(num_in_class[0]).tolist()   
        b_perm = [x + num_in_class[0] for x in torch.randperm(num_in_class[1]).tolist()]

        if num_in_class[0] > num_in_class[1]:
            a_perm = a_perm[0:num_in_class[1]]
        elif num_in_class[0] < num_in_class[1]:
            b_perm = b_perm[0:num_in_class[0]]

        assert len(a_perm) == len(b_perm)  
        
       
        imgdata=[]
        for i in range(len(a_perm)):
            imgdata.append(a_perm[i])
            imgdata.append(b_perm[i])

        #return iter(next(it) for it in itertools.cycle([iter(a_perm), iter(b_perm)]))
        return iter(imgdata)

    def __len__(self):
        return min(self.num_in_class) * 2


# each two element is paired, and order is shuffled for each epoch (shuffle=True)
# the number of samples in two class is same
class PairedSampler(Sampler):
    def __init__(self, data_source):
        print('Using PairedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = torch.randperm(num_in_class[0]).tolist()
        b_perm = [x + num_in_class[0] for x in a_perm]

        return iter(next(it) for it in itertools.cycle([iter(a_perm), iter(b_perm)]))

    def __len__(self):
        return min(self.num_in_class) * 2


# each two element is paired
# the number of samples in two class is same
class SequentialPairedSampler(Sampler):
    def __init__(self, data_source):
        print('Using SequentialPairedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        a_perm = range(num_in_class[0])
        b_perm = [x + num_in_class[0] for x in a_perm]

        return iter(next(it) for it in itertools.cycle([iter(a_perm), iter(b_perm)]))

    def __len__(self):
        return min(self.num_in_class) * 2


# each two element is paired
# the number of samples is times of that of other class
class PairedBalancedSampler(Sampler):
    def __init__(self, data_source):
        print('Using PairedBalancedSampler...')
        self.data_source = data_source
        self.num_in_class = data_source.num_in_class

    def __iter__(self):
        num_in_class = self.num_in_class
        if num_in_class[0] > num_in_class[1]:
            pn_num = num_in_class[0] / num_in_class[1]
            classA_index = np.arange(num_in_class[0]).reshape(num_in_class[1], -1)
            rad = np.random.rand(classA_index.shape[0], classA_index.shape[1])
            column_index = np.argmax(rad, axis=1)

            tmp_perm = torch.randperm(num_in_class[1])
            a_perm = list(classA_index[np.array(tmp_perm.tolist()), column_index])
            b_perm = [x + num_in_class[0] for x in tmp_perm]
        elif num_in_class[0] < num_in_class[1]:
            pn_num = num_in_class[1] / num_in_class[0]
            classB_index = np.arange(num_in_class[1]).reshape(num_in_class[0], -1)
            rad = np.random.rand(classB_index.shape[0], classB_index.shape[1])
            column_index = np.argmax(rad, axis=1)

            a_perm = torch.randperm(num_in_class[0])
            tmp_perm = list(classB_index[np.array(a_perm.tolist()), column_index])
            b_perm = [x + num_in_class[0] for x in tmp_perm]

        assert len(a_perm) == len(b_perm)

        return iter(next(it) for it in itertools.cycle([iter(a_perm), iter(b_perm)]))

    def __len__(self):
        return min(self.num_in_class) * 2


class DataLoaderHalf(DataLoader):
    def __init__(self, dataset,
                 shuffle=False, batch_size=1, half_constraint=False, sampler_type='RandomBalancedSampler',
                 drop_last=True, num_workers=0, pin_memory=False):
        if half_constraint:
            if sampler_type == 'PairedSampler' or sampler_type == 'SequentialPairedSampler':
                if shuffle:
                    sampler = PairedSampler(dataset)
                else:
                    sampler = SequentialPairedSampler(dataset)
            elif sampler_type == 'PairedBalancedSampler':
                sampler = PairedBalancedSampler(dataset)
            else:
                sampler = RandomBalancedSampler(dataset)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)   
            else:
                sampler = SequentialSampler(dataset)   

        super(DataLoaderHalf, self). \
            __init__(dataset, batch_size, None, sampler,
                     None, num_workers, pin_memory=pin_memory, drop_last=drop_last)

