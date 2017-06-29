import math
import os
import imagenet_utils
import tensorflow as tf
import numpy as np
from enum import Enum
from collections import namedtuple

from nets import nets_factory
from ImagenetModel import ImagenetModel

Preproc = Enum('Preprocessing', 'Inception VGG')

NetInfo = namedtuple('NetInfo',
                     ['name', 'preproc',
                      'resol', 'nb_class',
                      'default_model_fname'])

all_nets = [
        NetInfo('inception_resnet_v2', Preproc.Inception,
                (299, 299), 1001,
                'inception_resnet_v2_2016_08_30.ckpt'),
        NetInfo('resnet_v2_152', Preproc.Inception,
                (299, 299), 1001,
                'resnet_v2_152_2017_04_14.ckpt'),
        NetInfo('resnet_v1_152', Preproc.VGG,
                (224, 224), 1000,
                'resnet_v1_152_2016_08_28.ckpt'),
        NetInfo('resnet_v1_50', Preproc.VGG,
                (224, 224), 1000,
                'resnet_v1_50_2016_08_28.ckpt')
]
name_to_netinfo = dict(map(lambda x: (x.name, x), all_nets))

class SlimModel(ImagenetModel):
    def __init__(self, net_info, batch_size):
        self.RESOL = net_info.resol
        self.batch_size = batch_size
        self.net_info = net_info

        # Assign the preprocessing function
        if net_info.preproc == Preproc.Inception:
            self.preprocess_np = super(type(self), self).preprocess_inception
        elif net_info.preproc == Preproc.VGG:
            self.preprocess_np = super(type(self), self).preprocess_vgg
        else:
            raise Exception('Unsupported preprocessing')

    def preprocess_pil(self, img):
        return super(type(self), self).preprocess_pil(img)

    def raw_predict(self, imgs):
        num_batches = int(math.ceil(imgs.shape[0] / float(self.batch_size)))
        preds = []
        for i in xrange(num_batches):
            inds = range(i * self.batch_size, (i + 1) * self.batch_size)
            batch = imgs.take(inds, axis=0, mode='clip')
            batch_pred = self.sess.run(self.logits,
                                       feed_dict={self.images: batch})
            preds.append(batch_pred)
        preds = np.concatenate(preds, axis=0)
        preds = preds[0:imgs.shape[0], :]
        if len(preds.shape) > 2:
            preds = np.squeeze(preds, axis=0)
        return preds

    def predict_imagenet(self, imgs):
        preds = self.raw_predict(imgs)
        print preds.shape
        if preds.shape[1] == 1001:
            preds = preds[:, :-1]
        return imagenet_utils.decode_predictions(preds)

    def load_model(self, fname):
        slim = tf.contrib.slim

        self.graph = tf.get_default_graph()
        tf_global_step = slim.get_or_create_global_step()

        network_fn = nets_factory.get_network_fn(
                self.net_info.name,
                self.net_info.nb_class,
                is_training=False)

        self.images = tf.placeholder(
                tf.float32,
                shape=(self.batch_size, self.RESOL[0], self.RESOL[1], 3))
        tmp, _ = network_fn(self.images)
        self.logits = tf.nn.softmax(tmp)
        variables_to_restore = slim.get_variables_to_restore()
        self.sess = tf.Session(graph=self.graph)
        saver = tf.train.Saver()
        saver.restore(self.sess, fname)
