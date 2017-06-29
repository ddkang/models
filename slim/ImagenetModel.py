import abc
import keras.preprocessing.image

class ImagenetModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def load_model(self, fname):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess_pil(self, img):
        """Preprocess a SINGLE PIL image"""
        # ASSUMES RESOL IS SQUARE
        RESOL = self.RESOL[0]
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

    def preprocess_vgg(self, imgs):
        if imgs.shape[-3:-1] != self.RESOL or imgs.shape[-1] != 3:
            raise RuntimeError('Resolution wrong: ' + str(imgs.shape))
        imgs[:, :, :, 0] -= 123.68
        imgs[:, :, :, 1] -= 116.78
        imgs[:, :, :, 2] -= 103.94
        return imgs

    def preprocess_inception(self, imgs):
        if imgs.shape[-3:-1] != self.RESOL or imgs.shape[-1] != 3:
            raise RuntimeError('Resolution wrong: ' + str(imgs.shape))
        imgs /= 255.
        imgs -= 0.5
        imgs *= 2.
        return imgs

    # @abc.abstractmethod
    def preprocess_np(self, imgs):
        """Preprocess numpy array of images"""
        raise NotImplementedError

    @abc.abstractmethod
    def raw_predict(self, imgs):
        """Some of the nets have a "background" class, so this outputs the raw data"""
        raise NotImplementedError

