import numpy as np
import os
import torch
from torchvision import models, transforms
import object3d

class ModelNet40WithImage(torch.utils.data.Dataset):
    def __init__(self, cfg, renderer : object3d.Panda3DRenderer, mode='train'):
        self.root = cfg['data_root']
        self.augmentation = cfg['augmentation']
        self.mode = mode

        self.data = []
        labels = os.listdir(self.root)
        labels.sort()
        for label_index, label in enumerate(labels):
            type_root = os.path.join(self.root, label, mode)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz'):
                    identity = os.path.join(type_root, filename[:-4])
                    self.data.append((identity, label_index))
                    
        # init renderer
        self.renderer = renderer
        
        # resnet transform
        if mode == 'train':
            self.resnet_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, i):
        identity, label = self.data[i]
        
        # generate 2D Image
        self.renderer.set_obj(identity + '.off')
        self.renderer.random_context(coverage=0.8, obj_color=True, obj_translate=False, obj_rotation=True)
        pil_image = self.renderer.render(binary=False).convert('RGB')
        torch_image = self.resnet_transform(pil_image)
        
        # process meshnet data
        data = np.load(identity + '.npz')
        centers = data['centers'] # [face, 3]
        corners = data['corners'] # [face, vertice, 3]
        normals = data['normals'] # [face, 3]
        neighbors_index = data['neighbors_index'] # [face, ?]
        
        num_point = len(centers)
        # fill for n < 1024
        if num_point < 1024:
            chosen_indexes = np.random.randint(0, num_point, size=(1024 - num_point))
            centers = np.concatenate((centers, centers[chosen_indexes]))
            corners = np.concatenate((corners, corners[chosen_indexes]))
            normals = np.concatenate((normals, normals[chosen_indexes]))
            neighbors_index = np.concatenate((neighbors_index, neighbors_index[chosen_indexes]))
            
            # choose 3 neighbors
            new_neighbors_index = np.empty([1024, 3], dtype=np.int64)
            for idx in range(1024):
                neighbors = neighbors_index[idx]
                if len(neighbors) > 3:
                    new_neighbors_index[idx] = np.random.choice(neighbors, 3, replace=False)
                else:
                    new_neighbors_index[idx] = np.concatenate((neighbors, [idx]*(3-len(neighbors))))

            neighbors_index = new_neighbors_index
        else:
            chosen_indexes = np.random.choice(num_point, size=1024, replace=False)
            centers = centers[chosen_indexes]
            corners = corners[chosen_indexes]
            normals = normals[chosen_indexes]
            neighbors_index = neighbors_index[chosen_indexes]
            # remove unlinkable index and choose 3 neighbors
            new_neighbors_index = np.empty([1024, 3], dtype=np.int64)
            for idx in range(1024):
                mask = np.in1d(neighbors_index[idx], chosen_indexes)
                neighbors = np.array(neighbors_index[idx])[mask]
                if len(neighbors) > 3:
                    new_neighbors_index[idx] = np.random.choice(neighbors, 3, replace=False)
                else:
                    new_neighbors_index[idx] = np.concatenate((neighbors, [chosen_indexes[idx]]*(3-len(neighbors))))

            # re-index the neighbor
            invert_index = {value: key for key, value in enumerate(chosen_indexes)}
            neighbors_index = np.vectorize(invert_index.get, cache=True)(new_neighbors_index)

        # data augmentation
        if self.augmentation and self.mode == 'train':
            centers = self.__augment__(centers)
            corners = self.__augment__(corners)

        # make corner relative to center
        corners = corners - centers[:, np.newaxis, :]
        corners = corners.reshape([-1, 9])

        # to tensor
        centers = torch.from_numpy(centers).float().permute(1, 0).contiguous()
        corners = torch.from_numpy(corners).float().permute(1, 0).contiguous()
        normals = torch.from_numpy(normals).float().permute(1, 0).contiguous()
        neighbors_index = torch.from_numpy(neighbors_index).long()

        return torch_image, centers, corners, normals, neighbors_index

    def __augment__(self, data):
        sigma, clip = 0.01, 0.05
        jittered_data = np.clip(sigma * np.random.randn(*data.shape), -clip, clip)
        return data + jittered_data

    def __len__(self):
        return len(self.data)
    
    def close(self):
        self.renderer.close()

if __name__ == "__main__":
    dataset = ModelNet40WithImage({'data_root': '../meshnet_data/ModelNet40/', 'augment_data': True})
    dataset[0]