import os
import copy
import torch
from datetime import datetime
from collections import OrderedDict
from torchvision import models
from modules.identity import Identity
from models import MeshNet

class Image2ComEmbedding(torch.nn.Module):
    def __init__(self, cfg):
        super(Image2ComEmbedding, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = Identity()
        self.resnet = resnet
        
        self.embed_resnet = torch.nn.Sequential(OrderedDict([
            ('embed_resnet_fc1', torch.nn.Linear(2048, 512)),
            ('embed_resnet_relu1', torch.nn.ReLU6()),
            ('embed_resnet_dropout', torch.nn.Dropout(p=0.5)),
            ('embed_resnet_fc2', torch.nn.Linear(512, 1024)),
            ('embed_resnet_relu2', torch.nn.ReLU6()),
        ]))

        if not cfg['refine']:
            for pmt in self.resnet.parameters():
                pmt.requires_grad = False
                
    def forward(self, images):
        resnet_feature = self.resnet(images)
        embedded_image = self.embed_resnet(resnet_feature)
        
        return embedded_image
    
class Model3D2ComEmbedding(torch.nn.Module):
    def __init__(self, cfg):
        super(Model3D2ComEmbedding, self).__init__()
        self.meshnet = MeshNet(cfg, head=False)
        pretrained = cfg.get('pretrained', None)
        if pretrained:
            state_dict = torch.load(pretrained, 
                map_location=lambda storage, location: storage.cuda() if torch.cuda.is_available() else storage)
            self.meshnet.load_state_dict(state_dict, strict=False)
        
        self.embed_meshnet = torch.nn.Sequential(OrderedDict([
            ('embed_meshnet_fc1', torch.nn.Linear(256, 1280)),
            ('embed_resnet_relu1', torch.nn.ReLU6()),
            ('embed_meshnet_dropout', torch.nn.Dropout(p=0.5)),
            ('embed_meshnet_fc2', torch.nn.Linear(1280, 1024)),
            ('embed_resnet_relu2', torch.nn.ReLU6()),
        ]))
        
        if not cfg['refine']:
            for pmt in self.meshnet.parameters():
                pmt.requires_grad = False
            
    def forward(self, centers, corners, normals, neighbor_index):        
        meshnet_feature = self.meshnet(centers, corners, normals, neighbor_index)
        embedded_3dmodel = self.embed_meshnet(meshnet_feature)
        
        return embedded_3dmodel
    
class IMEmbedding(torch.jit.ScriptModule):
    def __init__(self, cfg):
        super(IMEmbedding, self).__init__()
        batch_size = cfg['batch_size']
        image_size = cfg['image_size']
        model3d_size = cfg['model3d_size']
        
        embed_image = Image2ComEmbedding(cfg['resnet'])
        self.embed_image = embed_image
        images = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
        embed_image.eval()
        self.embed_image_eval = torch.jit.trace(embed_image, [images], check_trace=False)
        embed_image.train()
        self.embed_image_train = torch.jit.trace(embed_image, [images], check_trace=False)
        
        embed_model3d = Model3D2ComEmbedding(cfg['meshnet'])
        self.embed_model3d = embed_model3d
        centers = torch.randn(1, 3, model3d_size, dtype=torch.float32)
        normals = torch.randn(1, 3, model3d_size, dtype=torch.float32)
        corners = torch.randn(1, 9, model3d_size, dtype=torch.float32)
        neighbor_index = torch.randint(0, model3d_size, (1, model3d_size, 3), dtype=torch.long)
        embed_model3d.eval()
        self.embed_model3d_eval = torch.jit.trace(embed_model3d, 
                                  [centers, corners, normals, neighbor_index], check_trace=False)
        embed_model3d.train()
        self.embed_model3d_train = torch.jit.trace(embed_model3d, 
                                   [centers, corners, normals, neighbor_index], check_trace=False)
        
        
    @torch.jit.script_method
    def __loss(self, images, model3d):
        # [image, model3d] or [batch, batch]
        im_similarity = torch.matmul(images, model3d.transpose(0, 1)) 
        # ground-truth
        sim_gt = torch.diagonal(im_similarity)
        # curremt match
        sorted_image, sarg_image = torch.sort(im_similarity, dim=1, descending=True)
        sorted_model3d, sarg_model3d = torch.sort(im_similarity, dim=0, descending=True)
        
        top1_image = sorted_image[:, 0]
        top1_model3d = sorted_model3d[0, :]
        
        # loss weights
        batch_size = im_similarity.shape[0]
        range_batch = torch.arange(0, batch_size, dtype=torch.long)
        grid_image, grid_model3d = torch.meshgrid(range_batch, range_batch)
        gt_image_rank = torch.nonzero(sarg_image == grid_image)[:, 1]
        gt_model3d_rank = torch.nonzero(sarg_model3d == grid_model3d)[:, 0]
        
        # combine loss with weights
        loss_image = torch.dot((top1_image - sim_gt), gt_image_rank.float() / batch_size)
        loss_model3d = torch.dot((top1_model3d - sim_gt), gt_model3d_rank.float() / batch_size)
        loss = (torch.sum(loss_image) + torch.sum(loss_model3d)) / batch_size
        
        return loss
    
    @torch.jit.script_method
    def forward(self, images, centers, corners, normals, neighbor_index):
        embedded_images = self.embed_image_train(images)
        embedded_model3d = self.embed_model3d_train(centers, corners, normals, neighbor_index)
        
        return embedded_images, embedded_model3d
    
    @torch.jit.script_method
    def forward_with_loss(self, images, centers, corners, normals, neighbor_index):
        embedded_images, embedded_model3d = self.forward(images, centers, corners, normals, neighbor_index)
        loss = self.__loss(embedded_images, embedded_model3d)
        return loss
    
    @torch.jit.script_method
    def forward_eval(self, images, centers, corners, normals, neighbor_index):
        embedded_images = self.embed_image_eval(images)
        embedded_model3d = self.embed_model3d_eval(centers, corners, normals, neighbor_index)
        
        return embedded_images, embedded_model3d
    
    @torch.jit.script_method
    def forward_with_loss_eval(self, images, centers, corners, normals, neighbor_index):
        embedded_images, embedded_model3d = self.forward_eval(images, centers, corners, normals, neighbor_index)
        loss = self.__loss(embedded_images, embedded_model3d)
        return loss
    
    def fit(self, dataloaders, optimizer=None, start_epoch=1, end_epochs=30, ckpt_root='ckpt_root'):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        val_loss_hist = []
        for epoch in range(start_epoch, end_epochs + 1):
            print('-' * 60)
            print('Epoch: {} / {}'.format(epoch, end_epochs))
            print('-' * 60)
            
            for phrase in ['train', 'val']:
                if phrase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                batch_size = dataloaders[phrase].batch_size
                dataset_size = len(dataloaders[phrase].dataset)
                total_steps = int(dataset_size / batch_size)
                
                for step, (images, centers, corners, normals, neighbor_index) in enumerate(dataloaders[phrase]):
                    optimizer.zero_grad()
                    
                    if torch.cuda.is_available():
                        images = images.cuda()
                        centers = centers.cuda()
                        normals = normals.cuda()
                        corners = corners.cuda()
                        neighbor_index = neighbor_index.cuda()
                        targets = targets.cuda()
                        
                    with torch.set_grad_enabled(phrase == 'train'):
                        
                        
                        if phrase == 'train':
                            loss = self.forward_with_loss(images, centers, corners, normals, neighbor_index)
                            loss.backward()
                            optimizer.step()
                        else:
                            loss = self.forward_with_loss_eval(images, centers, corners, normals, neighbor_index)
                            
                        batch_loss = loss.item()
                        running_loss += batch_loss * batch_size
                        
                    print(f'{datetime.now()} {phrase} epoch: {epoch}/{end_epochs} '
                          f'step:{step + 1}/{total_steps} loss: {batch_loss:.4f}')
                          
                epoch_loss = running_loss / dataset_size
                print(f'{phrase} epoch: {epoch}/{end_epochs} loss: {epoch_loss:.4f}')
                
                if phrase == 'train':
                    os.makedirs(ckpt_root, exist_ok=True)
                    filename = os.path.join(ckpt_root, f'{epoch:04d}.pkl')
                    torch.save(copy.deepcopy(self.state_dict()), filename)
                
                if phrase == 'val':
                    val_loss_hist.append(epoch_loss)
                    
        return val_loss_hist