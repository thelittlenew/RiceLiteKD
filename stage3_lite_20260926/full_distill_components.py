"""Frozen Refine7 source-distillation adapter; no additional trainable parameters."""
import hashlib,io,json,random
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from matched_cgnet_refine7_recipe import CGNetRefine7NativeSeg


class Refine7DistillStudent(CGNetRefine7NativeSeg):
    def forward(self,x):
        features=self.native.extract_feat(x)
        logits=self.native.decode_head(features)
        logits=F.interpolate(logits,size=x.shape[-2:],mode='bilinear',align_corners=False)
        # Parameter-free P2/P3/P4/P5 signals for channel-independent attention transfer.
        pyramid=[features[1],features[2],F.avg_pool2d(features[2],2),F.avg_pool2d(features[2],4)]
        return {'logits':logits,'features':pyramid}


class FullSourceDataset(Dataset):
    def __init__(self,manifest,training=False,epoch=1):
        self.rows=json.loads(Path(manifest).read_text())
        self.training,self.epoch=training,epoch
        self.handles=OrderedDict()
    def __len__(self):return len(self.rows)
    def read(self,row,kind):
        name=row['archive']
        if name not in self.handles:
            if len(self.handles)>=8:self.handles.popitem(last=False)[1].close()
            self.handles[name]=open(name,'rb')
        f=self.handles[name];self.handles.move_to_end(name)
        f.seek(row[kind+'_offset']);data=f.read(row[kind+'_bytes'])
        assert len(data)==row[kind+'_bytes']
        return data
    def __getitem__(self,i):
        row=self.rows[i]
        # cv2 decode matches the original source Stage3 RGB conversion.
        raw=np.frombuffer(self.read(row,'image'),dtype=np.uint8)
        rgb=cv2.cvtColor(cv2.imdecode(raw,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        mask=np.array(Image.open(io.BytesIO(self.read(row,'mask'))),dtype=np.uint8)
        assert rgb.shape==(512,512,3) and mask.shape==(512,512)
        if self.training:
            rng=random.Random(42+self.epoch*1000003+i)
            if rng.random()<.5:rgb,mask=rgb[:,::-1],mask[:,::-1]
            if rng.random()<.3:rgb,mask=rgb[::-1],mask[::-1]
            if rng.random()<.3:
                k=rng.randint(1,3);rgb,mask=np.rot90(rgb,k),np.rot90(mask,k)
            if rng.random()<.5:rgb=np.clip(rgb.astype(np.float32)*rng.uniform(.8,1.2)+rng.uniform(-15,15),0,255).astype(np.uint8)
            if rng.random()<.1:rgb=cv2.GaussianBlur(np.ascontiguousarray(rgb),(3,3),0)
        rgb=(rgb.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return torch.from_numpy(rgb.copy()).permute(2,0,1),torch.from_numpy(mask.astype(np.int64).copy())


def class_weights(inventory):
    counts=inventory['pixel_counts_train']
    x=np.array([counts['background'],counts['foreground']],dtype=np.float64)
    freq=x/x.sum();weights=np.clip(np.median(freq[freq>0])/freq,.2,5.)
    weights/=weights.mean()
    return torch.tensor(weights,dtype=torch.float32)


class FullDistillationLoss(nn.Module):
    def __init__(self,weights):
        super().__init__();self.register_buffer('weights',weights)
    def forward(self,student,teacher,target):
        logits=student['logits'].float();valid=target!=255;safe=target.masked_fill(~valid,0)
        ce=F.cross_entropy(logits,target,weight=self.weights,ignore_index=255)
        probs=logits.softmax(1)*valid[:,None]
        truth=F.one_hot(safe,2).permute(0,3,1,2).float()*valid[:,None]
        dice=1-((2*(probs*truth).sum((0,2,3))+1)/(probs.sum((0,2,3))+truth.sum((0,2,3))+1)).mean()
        tl=F.interpolate(teacher['logits'].detach().float(),size=logits.shape[-2:],mode='bilinear',align_corners=False)
        # Original batchmean sums all pixels; normalize per valid pixel to avoid resolution-dependent magnitude.
        kl=F.kl_div(F.log_softmax(logits/4.,1),F.softmax(tl/4.,1),reduction='none').sum(1)*16
        kd=kl[valid].mean()
        terms=[]
        for sf,tf,weight in zip(student['features'],teacher['features'],[1.,1.,1.,.5]):
            sa=sf.float().square().mean(1,keepdim=True)
            ta=tf.detach().float().square().mean(1,keepdim=True)
            sa=F.interpolate(sa,size=ta.shape[-2:],mode='bilinear',align_corners=False)
            mask=F.interpolate(valid[:,None].float(),size=ta.shape[-2:],mode='nearest')
            sa=F.normalize((sa*mask).flatten(1),dim=1,eps=1e-6)
            ta=F.normalize((ta*mask).flatten(1),dim=1,eps=1e-6)
            terms.append(weight*(sa-ta).square().sum(1).mean())
        feature=sum(terms)/3.5
        return {'total':ce+2*dice+2*kd+3*feature,'ce':ce,'dice':dice,'kd':kd,'feature':feature}
