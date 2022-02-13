import torch
import torchvision.transforms as transforms
import cv2 

from PIL import Image
import numpy as np 
import glob
import sys
sys.path.insert(1, 'ShelfNet18_realtime/')
from evaluate import MscEval
from shelfnet import ShelfNet

class auto_park_vision():
    def __init__(self,weights_path):

        self.n_classes = 19
        self.eval_define() #Define Object of class MscEval
        #self.evaluator is an object of the class MscEval

    def forward_pass(self,frame=None,img_path=None):

        if(img_path != None):
            img = Image.open(img_path)
        else:
            img = frame 
        orig_img = np.array(img)
        # orig_img = cv2.imread(img_path)
        # cv2_imshow(img)
        #Preprocess Image
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        img = to_tensor(img)
        

        _, H, W = img.shape
        # print("H,W:",H,W)
        #Change image size to the form NCHW from CHW
        img = img.unsqueeze(0)

        probs = torch.zeros((1, self.n_classes, H, W))
        probs.requires_grad = False
        img = img.cuda()        

        # for sc in self.scales:
        prob = self.evaluator.scale_crop_eval(img, scale=1.0) #prob.type torch.cuda.FloatTensor
        prob = prob.detach().cpu()
        prob = prob.data.numpy()
        preds = np.argmax(prob, axis=1) #preds.dtype int64
        # palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        # pred = palette[preds.squeeze()]

        #Changed 
        preds = preds.squeeze().astype(np.uint8)
        preds[preds == 0] = 255
        preds = preds.astype(np.uint8)
        return preds
        # overlay = np.copy(preds)
        # overlay = cv2.cvtColor(overlay,cv2.COLOR_GRAY2RGB)
        # cv2_imshow(preds)
        #Overlay preds over the original image
        # alpha = 0.5
        # cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha,0, orig_img)
        # cv2_imshow(orig_img)
        # orig_img = self.parking_spot_detection(preds,orig_img)
        # return orig_img


    def eval_define(self):

        n_classes = self.n_classes
        net = ShelfNet(n_classes=n_classes)

        net.load_state_dict(torch.load(weights_path))
        net.cuda()
        net.eval()
        self.evaluator = MscEval(net, dataloader=None, scales=[1.0],flip=False)

