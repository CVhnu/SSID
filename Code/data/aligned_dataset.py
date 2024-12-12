import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import random
import torchvision.transforms as transforms
import math

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.paths = sorted(make_dataset(os.path.join(opt.dataroot)))
        self.dataset_size = len(self.paths)

    def __getitem__(self, index):
        A_path = self.paths[index]
        clear = Image.open(A_path)
        size_w = clear.size[0]-clear.size[0] % 8
        size_h = clear.size[1]-clear.size[1] % 8
        new_size = [size_h, size_w]
        transform_A = get_transform(new_size, self.opt)
        A_tensor = transform_A(clear.convert('RGB'))
        if self.opt.isTrain:
            T = torch.ones(A_tensor.size(1), A_tensor.size(2))
            K = random.uniform(2.0, 6.0)
            KK = random.uniform(2.0, 6.0)
            b = 0
            S = random.randint(1, 9)
            M = random.uniform(0.2, 1.0)
            n = A_tensor.size(1)
            nn = A_tensor.size(2)
            if S == 1:
                for i in range(n):
                    T[(i), ...] = 1 / (1 + math.exp(-K * (i / A_tensor.size(1) - M))) + b  # test1

            elif S == 2:
                for i in range(n):
                    T[..., (i)] = 1 / (1 + math.exp(-K * (i / A_tensor.size(1) - M))) + b  # test2
            elif S == 3:
                for i in range(n):
                    T[..., (A_tensor.size(1) - i - 1)] = 1 / (1 + math.exp(-K * (i / A_tensor.size(1) - M))) + b  # test3
            elif S == 4:
                for i in range(n):
                    T[(A_tensor.size(1) - i - 1), ...] = 1 / (1 + math.exp(-K * (i / A_tensor.size(1) - M))) + b  # test4
            elif S == 5:
                for i in range(n + nn - 1):
                    for j in range(i + 1):
                        nnn = i - j
                        if nnn < n and nnn >= 0 and j < n:
                            T[j][nnn] = 1 / (1 + math.exp(-KK * (i / (2.0 * n) - M))) + b
            elif S == 6:
                for i in range(n + nn - 1):
                    for j in range(i + 1):
                        nnn = i - j
                        if nnn < n and nnn >= 0 and j < n:
                            T[n - 1 - nnn][n - j - 1] = 1 / (1 + math.exp(-KK * (i / (2.0 * n) - M))) + b
            elif S == 7:
                for i in range(n + nn - 1):
                    for j in range(i + 1):
                        nnn = i - j
                        if nnn < n and nnn >= 0 and j < n:
                            T[n - j - 1][nnn] = 1 / (1 + math.exp(-KK * (i / (2 * n) - M))) + b
            elif S == 8:
                for i in range(n + nn - 1):
                    for j in range(i + 1):
                        nnn = i - j
                        if nnn < n and nnn >= 0 and j < n:
                            T[nnn][n - j - 1] = 1 / (1 + math.exp(-KK * (i / (2 * n) - M))) + b
            else:
                T = torch.mul(T, random.uniform(0.1, 0.35))
            T = T.clamp(0.01, 0.35)
            haze_img_sin = A_tensor * T + random.uniform(0.6, 1.0) * (1 - T)
            transform_B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            haze_img_sin = transform_B(haze_img_sin)

            input_dict = {'clear': A_tensor, 'hazy': haze_img_sin}
        else:
            input_dict = {'clear': A_tensor, 'hazy': A_tensor}

        return input_dict

    def __len__(self):
        return len(self.paths) // self.opt.batchSize * self.opt.batchSize
