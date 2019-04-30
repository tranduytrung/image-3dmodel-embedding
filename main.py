import json
import torch
import object3d
from data.dataset import ModelNet40WithImage
from modules.embedding import IMEmbedding

def main():
    cfg = {
        'resnet': {
            'refine': False
        },
        'meshnet': {
            'refine': False,
            'structural_descriptor': {
                'num_kernel': 64,
                'sigma': 0.2
            },
            'mesh_convolution': {
                'aggregation_method': 'Concat'
            },
            'pretrained': './data/meshnet.pkl'
        },
        'batch_size': 4,
        'image_size': 224,
        'model3d_size': 1024,
        'data_root': './data/ModelNet40',
        'augmentation': True
    }

    renderer = object3d.Panda3DRenderer(output_size=(224, 224), 
                cast_shadow=True, light_on=True)
    loader_dict = {
        'train': torch.utils.data.DataLoader(ModelNet40WithImage(cfg, renderer, mode='train'), 4, num_workers=0,
                                            shuffle=True, pin_memory=True, drop_last=True),
        'val': torch.utils.data.DataLoader(ModelNet40WithImage(cfg, renderer, mode='test'), 4, num_workers=0,
                                            shuffle=True, pin_memory=True, drop_last=True),
    }

    embedding = IMEmbedding(cfg)
    val_hist = embedding.fit(loader_dict, end_epochs=2)
    with open('val_hist.txt', 'wt') as f:
        json.dump(val_hist, f)

if __name__ == "__main__":
    main()