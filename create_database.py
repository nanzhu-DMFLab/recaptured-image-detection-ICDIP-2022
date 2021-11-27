import os
import torchvision
from PIL import Image
import random

#os.system("find ./data -name '*.png' |xargs rm -rfv")

train_ratio=0.6
val_ratio=0.2
test_ratio=0.2

database_name='/home/nanzhu/Database/RID/RIDICASSP18'   # change to your database path
#type='RecapturedImages'
type='SingleCaptureImages'

type_path=os.path.join(database_name,type)

bksize=256

def central_cut_images(folder,target_path):
    for imgpath in folder:
        try:
            img=Image.open(imgpath)
            crop_obj=torchvision.transforms.CenterCrop((bksize,bksize))
            img=crop_obj(img)
            img.save(target_path+'/'+imgpath.split('/')[-2]+imgpath.split('/')[-1][0:-4]+'.png')
        except:
            print(imgpath)



IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.TIF','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


all_images=[]
for root, _, fnames in sorted(os.walk(type_path)):
    for fname in sorted(fnames):
        if is_image_file(fname):
            imgpath = os.path.join(root, fname)
            all_images.append(imgpath)
# shuffle
random.shuffle(all_images)
#split
len_all_images=len(all_images)
train_image=all_images[:int(train_ratio*len_all_images)]
val_image=all_images[int(train_ratio*len_all_images):int((train_ratio+val_ratio)*len_all_images)]
test_image=all_images[int((train_ratio+val_ratio)*len_all_images):]

print(len(train_image),len(val_image),len(test_image))

label= 'RI' if type=='RecapturedImages' else 'NI'
central_cut_images(train_image,os.path.join('./data/train_image',label))
central_cut_images(val_image,os.path.join('./data/val_image',label))
central_cut_images(test_image,os.path.join('./data/test_image',label))

