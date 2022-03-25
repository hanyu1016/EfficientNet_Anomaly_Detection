from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image 
from imutils import paths
import random
import csv
import os

def pil_loader(path):    # 一般採用pil_loader函式。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # return img
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customData(Dataset):  #在此做label
    def __init__(self,data_path, test_data_path, phase = '', data_transforms=None, loader = default_loader,is_shuffle=True,test_path=None):
        if phase == 'easy_val' or phase == 'test':
            img_paths = list(paths.list_images(test_data_path))
        else:    
            img_paths = list(paths.list_images(data_path))
        if is_shuffle:
            random.Random(4).shuffle(img_paths)

        self.img_path = []
        self.img_label = []

        endOfTrain = int(len(img_paths)*0.8)
        if phase =='train':
            phase_img_paths = img_paths[:endOfTrain] #取得list前6024筆
        elif phase == 'val':
            phase_img_paths = img_paths[endOfTrain:] #取得6024後的其他資料
        elif phase == 'easy_val':
            phase_img_paths = img_paths
            #Hanyu 
        elif phase=='test' and test_path!=None:
            phase_img_paths = list(paths.list_images(test_path))
            #print(phase_img_paths)
            #Hanyu
        if (test_path==None) and phase=='test':
            self.length=1
        else:
            self.length=len(phase_img_paths)
            for path in phase_img_paths:
                # if "OK" in path and "no_word" in path:
                if "OK" in path :
                    label = 0
                    self.img_path.append(path)
                    self.img_label.append(label)
                elif "NG" in path:
                    label = 1
                    self.img_path.append(path)
                    self.img_label.append(label)
                else:
                    label = -1
                    self.img_path.append(path)
                    self.img_label.append(label)

        self.data_transforms = data_transforms
        self.phase = phase
        self.loader = loader

    def __len__(self):
        # return len(self.img_path) 
        # return 300 #Hanyu
        return self.length

    def __getitem__(self, item):
        try:
            path = self.img_path[item]
            label = self.img_label[item]
            img = self.loader(path)

            if self.data_transforms is not None:
                try:
                    img = self.data_transforms[self.phase](img) #接到image
                except:
                    print("Cannot transform image: {}".format(path))
            
            if self.phase == 'easy_val':
                return path, img, label
            if self.phase == 'test':
                return path, img, label #Hanyu
            return img, label
        except:
            print(item)