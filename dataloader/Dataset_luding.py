import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py


class GeneralDataset(Dataset):
    def __init__(self, annotation_lines, in_shape, in_channels, num_classes, random, dataset_path): 
        self.annotation_lines = annotation_lines 
        self.path  = dataset_path    
        self.num_classes = num_classes 
        self.in_channels = in_channels  
        self.input_shape = in_shape    
        self.random = random      
        self.means = torch.tensor([0.2667404, 0.18832587, 0.22792856, 0.20986454, 0.24711227, 0.32796624, 0.32796624, 0.3365001, 0.35211313, 0.3654887, 0.3368311, 
                    0.30149728, 0.24235326, 0.4996434, 0.48073888, 0.47808385, 0.45176756, 0.63581246, 0.18859462, 0.53473216, 0.4583207, 0.49854693, 0.20224944])
        self.stds = torch.tensor([0.1460099, 0.10193796, 0.11544987, 0.11224613, 0.14937568, 0.18545857, 0.18545857, 0.19171631, 0.19602245, 0.19385922, 0.18811043, 
                    0.18651104, 0.1594925, 0.16230324, 0.22609118, 0.2465604, 0.040343847, 0.24212655, 0.10721293, 0.15048093, 0.18935789, 0.19029191, 0.11153882])
    
    def __len__(self):
        return len(self.annotation_lines) 

    def __getitem__(self, index):
        img, label = self.h5_ts(index)
        if self.random:
            img, label = self.ts_augmentation(img, label)
        tensor_img, tensor_label, tensor_oh_labels = self.process(img, label)

        return tensor_img, tensor_label, tensor_oh_labels

    def h5_ts(self, index):
        annotation_line = self.annotation_lines[index].split()[0] 

        self.name_img = annotation_line.split(',')[0].split('.')[0]
        self.name_label = annotation_line.split(',')[1].split('.')[0] 

        img_full_path = self.path+'/img/img.h5'
        label_full_path = self.path+'/label/label.h5'

        with h5py.File(img_full_path, "r") as img_h5:  
            img = img_h5[self.name_img][:]
        with h5py.File(label_full_path, "r") as label_h5: 
            label = label_h5[self.name_label][:]

        img = torch.tensor((img), dtype=torch.float32)
        label = torch.tensor((label))
        label[label>=self.num_classes] = 0

        min_value = torch.amin(img,(0,1))
        max_value = torch.amax(img,(0,1))
        img = torch.div((img - min_value), (max_value - min_value), out=torch.zeros_like(img)) 
        
        # img = torch.div((img-self.means), self.stds, out=torch.zeros_like(img)) 

        return img, label # hwc

    def random_crop(self, img, lbl, h, w):

        vw = torch.randint(low=0, high=w, size=(1,)).item() # valid corner point for the width
        vh = torch.randint(low=0, high=h, size=(1,)).item() # valid corner point for the height

        rw = w - vw # rest width to be removed
        rh = h - vh # rest height to be removed

        height, width, _ = img.shape
        img = img[vh:height-rh, vw:width-rw, :]
        lbl = lbl[vh:height-rh, vw:width-rw]
        img = img.permute(2,0,1) 
        img = F.interpolate(img.unsqueeze(0), size=(height, width), mode='bilinear').squeeze()
        lbl = F.interpolate(lbl.unsqueeze(0).unsqueeze(0), size=(height, width), mode='nearest').squeeze().squeeze()
        img = img.permute(1,2,0) 

        return img, lbl

    def random_erasing(self, img, p=0.5, sl=0.02, sh=0.2, r1=0.3):
        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1/r1)

        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

        if torch.rand(1).item() > self.p:
            return img

        else:
            while True:
                max = torch.max(img)
                Se = torch.zeros(1,)
                Se.uniform_(*self.s)
                Se = Se.item() * img.shape[0] * img.shape[1]

                re = torch.zeros(1,)
                re.uniform_(*self.r)
                re = re.item()
 
                He = int(round((Se * re)**0.5))
                We = int(round((Se / re)**0.5))

                xe = torch.randint(0, img.shape[1], size=(1,)).item()
                ye = torch.randint(0, img.shape[0], size=(1,)).item()

                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye : ye + He, xe : xe + We, :] = max*torch.rand(He, We, img.shape[2])

                    return img

    def ts_augmentation(self, image, label):#hwc
        H,_,_ = image.shape

        rotate = torch.randint(low=0, high=4, size=(1,)).item()
        if rotate == 0: 
            image = torch.rot90(image, k=1)
            label = torch.rot90(label, k=1)

        elif rotate == 1: 
            image = torch.rot90(image, k=2)
            label = torch.rot90(label, k=2)

        elif rotate == 2: 
            image = torch.rot90(image, k=3)
            label = torch.rot90(label, k=3)

        flip = torch.randint(low=0, high=3, size=(1,)).item()
        if flip == 0:
            image = torch.fliplr(image) 
            label = torch.fliplr(label) 

        elif flip == 1:
            image = torch.flipud(image)
            label = torch.flipud(label)

        crop = torch.rand(1).item()<.8
        if crop:
            h = torch.randint(low=1, high=H//3, size=(1,)).item()
            w = h
            image, label = self.random_crop(image, label, h, w)

        image = self.random_erasing(image,p=0.35)
            
        return image, label

    def process(self, in_img, label): 
        h, w = self.input_shape
        dh = in_img.shape[0]

        B01 = in_img[:,:,0:1]
        B02 = in_img[:,:,1:2]
        B03 = in_img[:,:,2:3]
        B04 = in_img[:,:,3:4]
        B05 = in_img[:,:,4:5]
        B06 = in_img[:,:,5:6]
        B07 = in_img[:,:,6:7]
        B08 = in_img[:,:,7:8]
        B8a = in_img[:,:,8:9]
        B09 = in_img[:,:,9:10]
        B10 = in_img[:,:,10:11]
        B11 = in_img[:,:,11:12]
        B12 = in_img[:,:,12:13]

        slope      = in_img[:,:,13:14]
        DEM        = in_img[:,:,14:15]
        aspect     = in_img[:,:,15:16]
        curv       = in_img[:,:,16:17]
        hillshade  = in_img[:,:,17:18]
        twi        = in_img[:,:,18:19]
        BSI        = in_img[:,:,19:20]
        NDVI       = in_img[:,:,20:21]
        NDWI       = in_img[:,:,21:22]
        brightness = in_img[:,:,22:23]

        B1to8 = in_img[:,:,0:8]
        B11to12 = in_img[:,:,11:13]
        

        img = torch.cat((B1to8,B09,B11to12,BSI,NDVI,NDWI,brightness,slope,DEM,aspect,curv,twi,hillshade), dim = -1)             #21c Paper Version
        # img = torch.cat((B1to8,B8a,B09,B11to12,BSI,NDVI,NDWI,brightness,slope,DEM,aspect,curv,twi,hillshade), dim = -1)       #21c + 8a
        # img = torch.cat((B1to8,B09,B10,B11to12,BSI,NDVI,NDWI,brightness,slope,DEM,aspect,curv,twi,hillshade), dim = -1)       #21c + b10
        # img = torch.cat((B1to8,B8a,B09,B10,B11to12,BSI,NDVI,NDWI,brightness,slope,DEM,aspect,curv,twi,hillshade), dim = -1)   #21c + 8a + b10
        # img = torch.cat((B02,B03,B04), dim = -1)                                             # 3c rgb
        # img = torch.cat((B1to8,B09,B11to12), dim = -1)                                       # 11c all spectra
        # img = torch.cat((B1to8,B09,B11to12,BSI,NDVI,NDWI,brightness), dim = -1)              # 11c+4c
        # img = torch.cat((B1to8,B09,B11to12,slope,DEM,aspect,curv,twi,hillshade), dim = -1)   # 11c+6c

        img = img.permute(2,0,1) 

        if h!=dh: 
            img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear').squeeze()
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze().squeeze()

        tensor_img = img
        tensor_label = label.long()         
        tensor_oh_labels = F.one_hot(tensor_label, self.num_classes+1)

        return tensor_img, tensor_label, tensor_oh_labels

