from model.scdunetpp import SCDUNetPP
from configs import cfg

def build_model(type):
    model = None
    if type =='scdunetpp':
        model =  SCDUNetPP(in_chans=cfg.dataloader.in_channels, num_class=cfg.model.num_classes)
    else:
        return ValueError

    return model