import os
import time
import feather
import numpy as np
import pandas as pd
from PIL import Image
from SlimModel import SlimModel, name_to_netinfo

# Core load and label
def load_batch(fnames, model):
    def load_image(fname):
        try:
            return Image.open(fname).convert('RGB')
        except:
            print 'Failed to load %s' % fname
            return Image.new('RGB', model.RESOL)
    pil_ims = map(load_image, fnames)
    imgs = map(lambda x: model.preprocess_pil(x), pil_ims)
    imgs = np.stack(imgs)
    imgs = model.preprocess_np(imgs)
    return imgs

def label(model, imgs):
    return model.raw_predict(imgs)

# Get image filenames
def get_fnames(csv_fname, id_to_path):
    df = pd.read_csv(csv_fname, sep='\t', names=['id', 'tags'])
    fnames = []
    for row in df.itertuples():
        if row.id in id_to_path:
            fnames.append(id_to_path[row.id])
    return fnames

# The IDs are given as id_hash.ext so we need a way to do id -> fname
def get_id_to_path(photo_path):
    id_to_path = {}
    for i in xrange(1000):
        cur_path = os.path.join(photo_path, '%03d' % i)
        fnames = os.listdir(cur_path)
        for fname in fnames:
            if fname[0] == '.':
                continue
            # Catch all for weird things
            try:
                cur_id = int(fname.split('_')[0])
                id_to_path[cur_id] = os.path.join(cur_path, fname)
            except:
                print fname
    return id_to_path



def main():
    BATCH_SIZE = 2000
    image_dir = '/lfs/1/ddkang/specializer/imagenet/ILSVRC2012_img_val/'
    fnames = os.listdir(image_dir)
    fnames = map(lambda x: os.path.join(image_dir, x), fnames)
    print len(fnames)

    model = SlimModel(name_to_netinfo['resnet_v1_50'], 100)
    model.load_model('weights/resnet_v1_50_2016_08_28.ckpt')

    df_all = []
    for i in range(0, len(fnames), BATCH_SIZE):
        tmp = fnames[i:i + BATCH_SIZE]
        batch = load_batch(tmp, model)
        begin = time.time()
        labels = label(model, batch)
        end = time.time()
        df = pd.DataFrame(labels)
        df.insert(0, 'fname', tmp)
        print df.shape
        print 'Finished batch %d out of %d, took %f s' % \
            (i, len(fnames) / BATCH_SIZE + 1, end - begin)
        df_all.append(df)
    df_all = pd.concat(df_all)
    print 'hi'
    print df_all.shape
    feather.write_dataframe(df_all, 'imagenet.feather')


if __name__ == '__main__':
    main()
