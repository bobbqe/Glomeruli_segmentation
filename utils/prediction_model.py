import torch
import torch.nn.functional as F

class Model_pred:
    def __init__(self, model, dl, cfg, tta: bool = False, half: bool = False):
        self.model = model
        self.dl = dl
        self.tta = tta
        self.half = half
        self.cfg = cfg

    def __iter__(self):
        self.model.eval()
        cfg = self.cfg
        name_list = self.dl.dataset.fnames
        count = 0
        with torch.no_grad():
            for x,y in iter(self.dl):
                x = x.to(cfg.MODEL.DEVICE)
                if self.half: x = x.half()
                p = self.model(x)
                py = torch.sigmoid(p).detach()
                if self.tta:
                    #x,y,xy flips as TTA
                    flips = [[-1],[-2],[-2,-1]]
                    for f in flips:
                        p = self.model(torch.flip(x,f))
                        p = torch.flip(p,f)
                        py += torch.sigmoid(p).detach()
                    py /= (1+len(flips))
                if y is not None and len(y.shape)==4 and py.shape != y.shape:
                    py = F.upsample(py, size=(y.shape[-2],y.shape[-1]), mode="bilinear")
                py = py.permute(0,2,3,1).float().cpu()
                batch_size = len(py)
                for i in range(batch_size):
                    taget = y[i].detach().cpu() if y is not None else None
                    yield py[i],taget,name_list[count]
                    count += 1
                    
    def __len__(self):
        return len(self.dl.dataset)

class Model_pred_test:
    def __init__(self, models, dl, cfg, half: bool = False):
        self.models = models
        self.dl = dl
        self.half = half
        self.cfg = cfg
        
    def __iter__(self):
        count=0
        with torch.no_grad():
            for x in iter(self.dl):
                x = x.to(self.cfg.MODEL.DEVICE)
                x = F.interpolate(x, scale_factor=1, mode='bilinear')
                if self.half: x = x.half()
                py = None
                for _,model in enumerate(self.models):
                    p = model(x)
                    p = torch.sigmoid(p).detach()
                    if py is None: 
                        py = p
                    else: py += p
                py /= len(self.models)
                    
                py = F.upsample(py, scale_factor=self.cfg.TEST.REDUCE, mode="bilinear")
                py = py.permute(0,2,3,1).float().cpu()
                py = (255*py).int()
                batch_size = len(py)
                for i in range(batch_size):
                    yield py[i]
                    count += 1
                    
    def __len__(self):
        return len(self.dl.dataset)
