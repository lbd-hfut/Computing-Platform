import torch,os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import scipy.io as sio

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lbdDataset(Dataset):
    def __init__(self, train_root):
        
        image_files = np.array([x.path for x in os.scandir(train_root)
                             if (x.name.endswith(".bmp") or
                             x.name.endswith(".png") or 
                             x.name.endswith(".JPG") or 
                             x.name.endswith(".tiff"))
                             ])
        image_files.sort()
        
        self.rfimage_files = [image_files[0]]
        self.mask_files = [image_files[-1]]
        self.dfimage_files = image_files[1:-1]
        
    
    def __len__(self):
        return len(self.dfimage_files)
    
    def __getitem__(self, idx):
        # Open images
        df_image = self.open_image(self.dfimage_files[idx])
        df_image = self.to_tensor(df_image)
        return df_image
                
    def open_image(self,name):
        img = Image.open(name).convert('L')
        img = np.array(img)
        return img
    
    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for to_tensor")
        
    def data_collect(self, device):
        rfimage = self.open_image(self.rfimage_files[0])
        mask = self.open_image(self.mask_files[0])
        unique_values = np.unique(mask) # Get the unique value in the mask
        if len(unique_values) == 2:
            # If there are only two values ??in the mask matrix
            mask = (mask > 0)  # Set values ??greater than 0 to 1, and the rest to 0
        else:
            # If there are multiple values ??in the mask matrix
            mask = (mask == 255)  # Set the value equal to 255 to 1, and the rest to 0
            
        RG = self.to_tensor(rfimage).to(device)
        ROI = self.to_tensor(mask).to(device)
        
        H,L = rfimage.shape
        y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
        IX, IY = np.meshgrid(x, y); IX = self.to_tensor(IX); IY = self.to_tensor(IY)
        XY = torch.stack((IX, IY), dim=2).unsqueeze(0).to(device)
        XY_roi = np.column_stack(np.where(mask == 1))
        XY_roi = torch.tensor(XY_roi).to(device)
        Ixy = torch.zeros_like(XY_roi); Ixy = Ixy.float() 
        Ixy[:,0] = 2 * (XY_roi[:, 1] - XY_roi[:, 1].min()) / \
            (XY_roi[:, 1].max() - XY_roi[:, 1].min()) - 1
        Ixy[:,1] = 2 * (XY_roi[:, 0] - XY_roi[:, 0].min()) / \
            (XY_roi[:, 0].max() - XY_roi[:, 0].min()) - 1
        Ixy = Ixy.to(device)
        return RG, ROI, XY, XY_roi, Ixy
    
class pyramidDataset(Dataset):
    def __init__(self, train_root):
        
        image_files = np.array([x.path for x in os.scandir(train_root)
                             if (x.name.endswith(".bmp") or
                             x.name.endswith(".png") or 
                             x.name.endswith(".JPG") or 
                             x.name.endswith(".tiff"))
                             ])
        image_files.sort()
        
        self.rfimage_files = [image_files[0]]
        self.mask_files = [image_files[-1]]
        self.dfimage_files = image_files[1:-1]
        
        # self.mask = self.open_image(self.mask_files[0])
        # unique_values = np.unique(self.mask) # Get the unique value in the mask
        # if len(unique_values) == 2:
        #     # If there are only two values ??in the mask matrix
        #     self.mask = (self.mask > 0)  # Set values ??greater than 0 to 1, and the rest to 0
        # else:
        #     # If there are multiple values ??in the mask matrix
        #     self.mask = (self.mask == 255)  # Set the value equal to 255 to 1, and the rest to 0
    
    def __len__(self):
        return len(self.dfimage_files)
    
    def __getitem__(self, idx):
        # Open images
        df_image = self.open_image(self.dfimage_files[idx])
        return df_image
                
    def open_image(self,name):
        img = Image.open(name).convert('L')
        img = np.array(img)
        return img
    
    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            return torch.tensor(array, dtype=torch.float32)
        elif isinstance(array, (int, float)):
            return torch.tensor([array], dtype=torch.float32)
        else:
            raise TypeError("Unsupported type for to_tensor")
        
    def data_collect(self, scale_factor, device):
        ref_img = self.open_image(self.rfimage_files[0])
        roi_img = self.open_image(self.mask_files[0])
        unique_values = np.unique(roi_img) # Get the unique value in the mask
        if len(unique_values) == 2:
            # If there are only two values ??in the mask matrix
            roi_img = (roi_img > 0)  # Set values ??greater than 0 to 1, and the rest to 0
        else:
            # If there are multiple values ??in the mask matrix
            roi_img = (roi_img == 255)  # Set the value equal to 255 to 1, and the rest to 0
        target_height = ref_img.shape[0] // scale_factor
        target_width  = ref_img.shape[1] // scale_factor
        
        down_ref = np.array(Image.fromarray(ref_img).resize((target_width, target_height), Image.NEAREST))
        down_roi = np.array(Image.fromarray(roi_img).resize((target_width, target_height), Image.NEAREST))
  
        RG  = self.to_tensor(down_ref).to(device)
        ROI = self.to_tensor(down_roi).to(device)
        ROI = ROI > 0
        
        H,L = RG.shape
        y = np.linspace(-1, 1, H); x = np.linspace(-1, 1, L); 
        IX, IY = np.meshgrid(x, y); IX = self.to_tensor(IX); IY = self.to_tensor(IY)
        XY = torch.stack((IX, IY), dim=2).unsqueeze(0).to(device)
        XY_roi = np.column_stack(np.where(down_roi > 0))
        XY_roi = torch.tensor(XY_roi).to(device)
        Ixy = torch.zeros_like(XY_roi); Ixy = Ixy.float() 
        Ixy[:,0] = 2 * (XY_roi[:, 1] - XY_roi[:, 1].min()) / \
            (XY_roi[:, 1].max() - XY_roi[:, 1].min()) - 1
        Ixy[:,1] = 2 * (XY_roi[:, 0] - XY_roi[:, 0].min()) / \
            (XY_roi[:, 0].max() - XY_roi[:, 0].min()) - 1
        Ixy = Ixy.to(device)
        return RG, ROI, XY, XY_roi, Ixy

    def down_sample(self, def_img, scale_factor, device):
        target_height = def_img.shape[0] // scale_factor
        target_width  = def_img.shape[1] // scale_factor
        
        down_def = np.array(Image.fromarray(def_img).resize((target_width, target_height), Image.NEAREST))
        DG = self.to_tensor(down_def).to(device)
        return DG
        
def collate_fn(batch):
    return batch  
        
    
    
# train_dataset = lbdDataset(train_root='../data/train')
# train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=1, shuffle=True,
#         collate_fn=collate_fn)

# data_iter = iter(train_loader)
# args = next(data_iter)
# a = train_dataset[0]
# for i, df_image in enumerate(train_loader):
#     print(i)
#     print(df_image.shape)
