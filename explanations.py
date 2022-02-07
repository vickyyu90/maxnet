import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm


class RISE(nn.Module):
    """A RISE class that computes saliency maps with RISE.

    """
    def __init__(self, model, input_size, N, p1, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.N = N
        self.p1 = p1

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            z = np.random.randint(0, cell_size[2])
            # Linear upsampling and cropping
            self.masks[i, :, :, :] = resize(grid[i], up_size, order=1, mode='reflect',anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1], z:z + self.input_size[2]]
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, L, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)
        stack = torch.unsqueeze(stack, 1)
        stack = stack.to(torch.float32)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            pred, _, _, _ = self.model(stack[i:min(i + self.gpu_batch, N)])
            p.append(nn.Softmax(dim=1)(pred))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W * L))
        sal = sal.view((CL, L, H, W))
        sal = sal / N / self.p1
        return sal

