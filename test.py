import os

import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from fire import Fire
from scripts.efficientTransformerSR import EfficientTransformerSR
from torchvision import transforms
from pathlib import Path
from scripts.dataloader import Set5Dataset, Set14Dataset
import tqdm
from skimage.metrics import structural_similarity


def forward_chop(model, x, shave=10, min_size=20000):
    scale = 2
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    if h_size % 2 != 0:
        h_size += 1
    if w_size % 2 != 0:
        w_size += 1

    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def apply_shave(dim_max, shave, x_start, x_end):
    if x_start - shave < 0:
        x_end += shave
    else:
        x_start -= shave
    if x_end + shave > dim_max:
        x_start -= shave
    else:
        x_end += shave
    return x_start, x_end


def forward_image(model, x, device, piece_size=48, scale_factor=2, shave=10):
    b, c, h, w = x.size()
    tiles = []

    for i in range(0, h, piece_size):
        for j in range(0, w, piece_size):
            if j + piece_size < w:
                w_start = j
                w_end = j + piece_size
            else:
                w_start = w - piece_size
                w_end = w
            if i + piece_size < h:
                h_start = i
                h_end = i + piece_size
            else:
                h_start = h - piece_size
                h_end = h

            w_start -= shave
            w_start = max(0, w_start)
            w_end += shave
            w_end = min(w, w_end)

            h_start -= shave
            h_start = max(0, h_start)
            h_end +=shave
            h_end = min(h, h_end)

            tmp = h_start - h_end
            if (h_start - h_end) % 2 != 0:
                h_start = h_start - 1

            if (w_start - w_end) % 2 != 0:
                w_start = w_start - 1

            tile = x[:, :, h_start:h_end, w_start:w_end]
            tiles.append(tile)

    sr_tiles = []

    for tile in tiles:
        tile = tile.to(device)
        sr_tile_tensor = torch.squeeze(model(tile))
        sr_tile = sr_tile_tensor.cpu().detach().numpy()
        sr_tiles.append(sr_tile)

    sr_image = np.empty(shape=[3, h*scale_factor, w*scale_factor])

    idx = 0
    scaled_shave = shave * scale_factor
    scaled_piece_size = piece_size * scale_factor

    for i in range(0, h, piece_size):
        for j in range(0, w, piece_size):
            if j + piece_size < w:
                w_start = j
                w_end = j + piece_size
            else:
                w_start = w - piece_size
                w_end = w
            if i + piece_size < h:
                h_start = i
                h_end = i + piece_size
            else:
                h_start = h - piece_size
                h_end = h

            sr_tile = sr_tiles[idx]

            if (max(0, (h_start - shave)) - min(h, h_end + shave)) % 2 != 0:
                sr_tile = sr_tile[:, scale_factor:, :]

            if (max(0, (w_start - shave)) - min(w, w_end + shave)) % 2 != 0:
                sr_tile = sr_tile[:, :, scale_factor:]

            w_start = w_start * scale_factor
            w_end = w_end * scale_factor
            h_start = h_start * scale_factor
            h_end = h_end * scale_factor

            if h_start > 0:
                sr_tile = sr_tile[:, scaled_shave:, :]

            if w_start > 0:
                sr_tile = sr_tile[:, :, scaled_shave:]
            h_start = max(h_start, 0)
            w_start = max(w_start, 0)

            sr_image[:, h_start:h_end, w_start:w_end] = sr_tile[:, :scaled_piece_size, :scaled_piece_size]
            idx = idx + 1

    return sr_image


def main(config = "test"):
    piece_size = 96
    scale_factor = 4

    app = EfficientTransformerSR(config)
    app.load()
    best_model = app.model
    best_model.eval()
    device = app.device
    test_dir = "tests" + f'\\X{scale_factor}' + "\\" + app.getModelName() + '\\Set14' + '\\' + f'piece-size-{piece_size}'

    dataset = Set14Dataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        lr_scale=scale_factor
    )

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    idx = 0
    #names = ['baby', 'bird', 'butterfly', 'head', 'woman']
    names = ['img_001', 'img_002', 'img_005', 'img_011', 'img_014']
    for img, label in dataset:
        file = names[idx] + '.png'
        x = torch.unsqueeze(img,0)
        x = x.to(device)

        label = np.clip(torch.squeeze(label).cpu().detach().numpy(), 0, 1)
        label = ((label * 255) / np.max(label)).astype(np.uint8)

        output = forward_image(best_model, x, device, piece_size=piece_size, scale_factor=scale_factor)
        output = np.clip(output, 0, 1)
        output = ((output * 255) / np.max(output)).astype(np.uint8)

        psnr = cv2.PSNR(output, label)
        print(f"Wartosc PSNR dla obrazu {file}, wynosi: {psnr}")
        test = np.moveaxis(output, 0, -1)

        img = Image.fromarray(test, 'RGB')
        img.save(test_dir + '\\' + file)
        idx = idx + 1

    '''
    for file in tqdm.tqdm(os.listdir(images_path)):
        image = Image.open(images_path + '\\' + file)
        convert_tensor = transforms.ToTensor()
        x = torch.unsqueeze(convert_tensor(image),0)
        x = x.to(device)

        output = forward_image(best_model, x, device, piece_size=piece_size, scale_factor=scale_factor)
        output = np.clip(output, 0, 1)
        output = ((output * 255) / np.max(output)).astype(np.uint8)
        test = np.moveaxis(output, 0, -1)

        img = Image.fromarray(test, 'RGB')
        img.save(test_dir + '\\' + file)
    '''
if __name__ == '__main__':
    Fire(main)