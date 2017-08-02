import functools
import os
import time
import feather
import keras
import tqdm
import multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image
from SlimModel import SlimModel, name_to_netinfo

def get_fnames(DIR_NUM):
    path = '/lfs/1/ddkang/specializer/yfcc100m/photos/%3d/' % DIR_NUM
    fnames = os.listdir(path)
    fnames.sort()
    fnames = map(lambda x: path + x, fnames)
    return fnames

def load_single_image(fname):
    RESOL = 224
    def preprocess_pil(img, RESOL=224):
        # ASSUMES RESOL IS SQUARE
        def get_central(x, y):
            def side(z):
                offset = RESOL % 2
                return (z/2 - RESOL/2, z/2 + RESOL/2 + offset)
            l, r = side(x)
            up, lo = side(y)
            return (l, up, r, lo)

        def central_crop(img):
            img = img.convert('RGB')
            scale = float(RESOL) / min(img.size)
            img = img.resize(map(lambda x: max(RESOL, int(x * scale)), img.size))
            img = img.crop(get_central(*img.size))
            return img

        im = img.convert('RGB')
        im = central_crop(im)
        return keras.preprocessing.image.img_to_array(im)
    def preproc_np(imgs):
        imgs[:, :, 0] -= 123.68
        imgs[:, :, 1] -= 116.78
        imgs[:, :, 2] -= 103.94
        return imgs
    def load_image(fname):
        try:
            return Image.open(fname).convert('RGB')
        except:
            print 'Failed to load %s' % fname
            return Image.new('RGB', (RESOL, RESOL))
    pil_im = load_image(fname)
    img = preprocess_pil(pil_im)
    img = preproc_np(img)
    return fname, img

def label(model, imgs):
    return model.raw_predict(imgs)

def main():
    DIR_NUM = 500
    BATCH_SIZE = 100

    fnames = get_fnames(DIR_NUM)[0:10000]

    model = SlimModel(name_to_netinfo['resnet_v1_152'], 100)
    model.load_model('weights/resnet_v1_152_2016_08_28.ckpt')

    pool = mp.Pool(10)
    imgs = []
    batch_fnames = []
    df_all = []
    for fname, im in tqdm.tqdm(pool.imap_unordered(load_single_image, fnames), total=len(fnames)):
        imgs.append(im)
        batch_fnames.append(os.path.basename(fname))
        if len(imgs) == BATCH_SIZE:
            batch = np.stack(imgs)
            labels = label(model, batch)
            df = pd.DataFrame(labels)
            df.insert(0, 'fname', batch_fnames)
            df_all.append(df)
            imgs = []
            batch_fnames = []
    df_all = pd.concat(df_all)
    print df_all.shape
    feather.write_dataframe(df_all, '%3d.feather' % DIR_NUM)

if __name__ == '__main__':
    main()
