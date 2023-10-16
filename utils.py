import os
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import json
import numpy as np

import math
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
irange=range


def save_image_ori(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, fp, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)





# def plot_images(images):
#     plt.figure(figsize=(32, 32))
#     plt.imshow(torch.cat([
#         torch.cat([i for i in images.cpu()], dim=-1),
#     ], dim=-2).permute(1, 2, 0).cpu())
#     plt.show()


# def save_images(images, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)


default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])


def resize_with_padding(img, expected_size, fill=0):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=fill)

def objects2onehot(labels):
    f = open('dataset/objects.json')
    datas = json.load(f)
    new_label = []
    for label in labels:
        items = []
        for item in label:
            items.append(datas[item])
        new_label.append(items)
    num_classes = 24
    onehot = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        for j in new_label[i]:
            onehot[i,j] = 1.
    return onehot


class iclevr_data(Dataset):
    def __init__(self, mode, transform = default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'new_test'

        self.mode = mode
        self.transform = transform
        self.dirs = []
        self.label = []
        if self.mode == 'train':
            f = open('dataset/train.json')

            datas = json.load(f)
            for item in datas.keys():
                self.dirs.append(item)

            for label in datas.values():
                self.label.append(label)

        elif self.mode == 'test':
            f = open('dataset/test.json')

            datas = json.load(f)
            for data in datas:
                self.label.append(data)

        elif self.mode == 'new_test':
            f = open('dataset/new_test.json')

            datas = json.load(f)
            for data in datas:
                self.label.append(data)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        labels = objects2onehot(self.label)
        label = torch.tensor(labels[index]).long()
        #label = torch.tensor(self.label[index])

        if self.mode == 'train':
            img = Image.open(os.path.join('iclevr',self.dirs[index])).convert('RGB')
            # padding : dimgray
            img = resize_with_padding(img, (320, 320), (105, 105, 105))
            image = self.transform(img)

            return image, label

        else:
            return label
        

def get_data(args):

    train_data = iclevr_data(mode='train')
    test_data = iclevr_data(mode='test')
    new_test_data = iclevr_data(mode='new_test')

    train_loader = DataLoader(train_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last= True,
                            pin_memory=True)

    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True)
    new_test_loader = DataLoader(new_test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True)

    return train_loader,test_loader,new_test_loader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
