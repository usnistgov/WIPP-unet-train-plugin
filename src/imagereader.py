# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    print('Python3 required')
    sys.exit(1)

import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import lmdb
import numpy as np
import os
import skimage.io
import skimage.transform
import scipy
import scipy.ndimage
import scipy.signal

from isg_ai_pb2 import ImageMaskPair
import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    import warnings
    warnings.warn('Codebase designed for Tensorflow 2.x.x')
import unet_model


def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    std = np.std(image_data)
    mv = np.mean(image_data)
    if std <= 1.0:
        # normalize (but dont divide by zero)
        image_data = (image_data - mv)
    else:
        # z-score normalize
        image_data = (image_data - mv) / std

    return image_data


def augment_image(img, mask=None, rotation_flag=0, reflection_flag=0,
                  jitter_augmentation_severity=0,  # jitter augmentation severity as a fraction of the image size
                  noise_augmentation_severity=0,  # noise augmentation as a percentage of current noise
                  scale_augmentation_severity=0,  # scale augmentation as a percentage of the image size):
                  blur_augmentation_max_sigma=0,  # blur augmentation kernel maximum size):
                  intensity_augmentation_severity=0):  # intensity augmentation as a percentage of the current intensity

    img = np.asarray(img)

    # ensure input images are np arrays
    img = np.asarray(img, dtype=np.float32)

    debug_worst_possible_transformation = False # useful for debuging how bad images can get

    # check that the input image and mask are 2D images
    assert len(img.shape) == 2 or len(img.shape) == 3

    # convert input Nones to expected
    if jitter_augmentation_severity is None:
        jitter_augmentation_severity = 0
    if noise_augmentation_severity is None:
        noise_augmentation_severity = 0
    if scale_augmentation_severity is None:
        scale_augmentation_severity = 0
    if blur_augmentation_max_sigma is None:
        blur_augmentation_max_sigma = 0
    if intensity_augmentation_severity is None:
        intensity_augmentation_severity = 0

    # confirm that severity is a float between [0,1]
    assert 0 <= jitter_augmentation_severity < 1
    assert 0 <= noise_augmentation_severity < 1
    assert 0 <= scale_augmentation_severity < 1
    assert 0 <= intensity_augmentation_severity < 1

    # get the size of the input image
    h, w, c = img.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        assert len(mask.shape) == 2 or len(mask.shape) == 3
        assert (mask.shape[0] == h and mask.shape[1] == w)

    # set default augmentation parameter values (which correspond to no transformation)
    orientation = 0
    reflect_x = False
    reflect_y = False
    jitter_x = 0
    jitter_y = 0
    scale_x = 1
    scale_y = 1

    if rotation_flag:
        orientation = 360 * np.random.rand()
    if reflection_flag:
        reflect_x = np.random.rand() > 0.5  # Bernoulli
        reflect_y = np.random.rand() > 0.5  # Bernoulli
    if jitter_augmentation_severity > 0:
        if debug_worst_possible_transformation:
            jitter_x = int(jitter_augmentation_severity * (w * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * 1))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y
        else:
            jitter_x = int(jitter_augmentation_severity * (w * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_x = -1 * jitter_x

            jitter_y = int(jitter_augmentation_severity * (h * np.random.rand()))  # uniform random jitter integer
            sign_val = np.random.rand() > 0.5  # Bernoulli
            if sign_val:
                jitter_y = -1 * jitter_y

    if scale_augmentation_severity > 0:
        max_val = 1 + scale_augmentation_severity
        min_val = 1 - scale_augmentation_severity
        if debug_worst_possible_transformation:
            scale_x = min_val + (max_val - min_val) * 1
            scale_y = min_val + (max_val - min_val) * 1
        else:
            scale_x = min_val + (max_val - min_val) * np.random.rand()
            scale_y = min_val + (max_val - min_val) * np.random.rand()

    # apply the affine transformation
    img = apply_affine_transformation(img, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)
    if mask is not None:
        mask = apply_affine_transformation(mask, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y)

    # apply augmentations
    if noise_augmentation_severity > 0:
        sigma_max = noise_augmentation_severity * (np.max(img) - np.min(img))
        max_val = sigma_max
        min_val = -sigma_max
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        sigma_img = np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * sigma
        img = img + sigma_img

    # apply blur augmentation
    if blur_augmentation_max_sigma > 0:
        max_val = blur_augmentation_max_sigma
        min_val = -blur_augmentation_max_sigma
        if debug_worst_possible_transformation:
            sigma = min_val + (max_val - min_val) * 1
        else:
            sigma = min_val + (max_val - min_val) * np.random.rand()
        if sigma < 0:
            sigma = 0
        if sigma > 0:
            img = scipy.ndimage.filters.gaussian_filter(img, sigma, mode='reflect')

    if intensity_augmentation_severity > 0:
        img_range = np.max(img) - np.min(img)
        if debug_worst_possible_transformation:
            value = 1 * intensity_augmentation_severity * img_range
        else:
            value = np.random.rand() * intensity_augmentation_severity * img_range
        if np.random.rand() > 0.5:
            sign = 1.0
        else:
            sign = -1.0
        delta = sign * value
        img = img + delta # additive intensity adjustment

    img = np.asarray(img, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        mask = np.round(mask)
        return img, mask
    else:
        return img


def apply_affine_transformation(I, orientation, reflect_x, reflect_y, jitter_x, jitter_y, scale_x, scale_y):

    if orientation is not 0:
        I = skimage.transform.rotate(I, orientation, preserve_range=True, mode='reflect')

    tform = skimage.transform.AffineTransform(translation=(jitter_x, jitter_y),
                                              scale=(scale_x, scale_y))
    I = skimage.transform.warp(I, tform._inv_matrix, mode='reflect', preserve_range=True)

    if reflect_x:
        I = np.fliplr(I)
    if reflect_y:
        I = np.flipud(I)

    return I



class ImageReader:

    def __init__(self, img_db, use_augmentation=True, balance_classes=False, shuffle=True, num_workers=1, number_classes=2, augmentation_reflection=0, augmentation_rotation=0, augmentation_jitter=0, augmentation_noise=0, augmentation_scale=0, augmentation_blur_max_sigma=0):
        random.seed()

        # copy inputs to class variables
        self.image_db = img_db
        self.use_augmentation = use_augmentation
        self.balance_classes = balance_classes
        self.shuffle = shuffle
        self.nb_workers = num_workers
        self.nb_classes = number_classes

        self._reflection_flag = augmentation_reflection
        self._rotation_flag = augmentation_rotation
        self._jitter_augmentation_severity = augmentation_jitter
        self._noise_augmentation_severity = augmentation_noise
        self._scale_augmentation_severity = augmentation_scale
        self._blur_max_sigma = augmentation_blur_max_sigma

        # init class state
        self.queue_starvation = False
        self.maxOutQSize = num_workers * 100 # queue 100 images per reader
        self.workers = None
        self.done = False

        # setup queue mechanism
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        # confirm that the input database exists
        if not os.path.exists(self.image_db):
            print('Could not load database file: ')
            print(self.image_db)
            raise IOError("Missing Database")

        # get a list of keys from the lmdb
        self.keys_flat = list()
        self.keys = list()
        for i in range(self.nb_classes):
            self.keys.append(list())

        self.lmdb_env = lmdb.open(self.image_db, map_size=int(2e10), readonly=True) # 20 GB
        self.lmdb_txns = list()

        datum = ImageMaskPair()  # create a datum for decoding serialized protobuf objects
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()

            # move cursor to the first element
            cursor.first()
            # get the first serialized value from the database and convert from serialized representation
            datum.ParseFromString(cursor.value())
            # record the image size
            self.image_size = [datum.img_height, datum.img_width, datum.channels]

            if self.image_size[0] % unet_model.UNet.SIZE_FACTOR != 0:
                raise IOError('Input Image tile height needs to be a multiple of 16 to allow integer sized downscaled feature maps')
            if self.image_size[1] % unet_model.UNet.SIZE_FACTOR != 0:
                raise IOError('Input Image tile height needs to be a multiple of 16 to allow integer sized downscaled feature maps')

            # iterate over the database getting the keys
            for key, val in cursor:
                self.keys_flat.append(key)

                if self.balance_classes:
                    datum.ParseFromString(val)
                    # get list of classes the current sample has
                    # convert from string to numpy array
                    cur_labels = np.fromstring(datum.labels, dtype=datum.mask_type)
                    # walk through the list of labels, adding that image to each label bin
                    for l in cur_labels:
                        self.keys[l].append(key)

        print('Dataset has {} examples'.format(len(self.keys_flat)))
        if self.balance_classes:
            print('Dataset Example Count by Class:')
            for i in range(len(self.keys)):
                print('  class: {} count: {}'.format(i, len(self.keys[i])))

    def get_image_count(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat))

    def get_image_size(self):
        return self.image_size

    def get_image_tensor_shape(self):
        # HWC to CHW
        return [self.image_size[2], self.image_size[0], self.image_size[1]]

    def get_label_tensor_shape(self):
        return [self.image_size[0], self.image_size[1]]

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        [self.lmdb_txns.append(self.lmdb_env.begin(write=False)) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        # wait for the workers to terminate
        for w in self.workers:
            w.join()

    def __get_next_key(self):
        if self.shuffle:
            if self.balance_classes:
                # select a class to add at random from the set of classes
                label_idx = random.randint(0, self.nb_classes - 1)  # randint has inclusive endpoints
                # randomly select an example from the database of the required label
                nb_examples = len(self.keys[label_idx])
                
                while nb_examples == 0:
                    # select a class to add at random from the set of classes
                    label_idx = random.randint(0, self.nb_classes - 1)  # randint has inclusive endpoints
                    # randomly select an example from the database of the required label
                    nb_examples = len(self.keys[label_idx])

                img_idx = random.randint(0, nb_examples - 1)
                # lookup the database key for loading the image data
                fn = self.keys[label_idx][img_idx]
            else:
                # select a key at random from the list (does not account for class imbalance)
                fn = self.keys_flat[random.randint(0, len(self.keys_flat) - 1)]
        else:  # no shuffle
            # without shuffle you cannot balance classes
            fn = self.keys_flat[self.key_idx]
            self.key_idx += self.nb_workers
            self.key_idx = self.key_idx % len(self.keys_flat)

        return fn

    def __image_loader(self):
        termimation_flag = False  # flag to control the worker shutdown
        self.key_idx = self.idQ.get()  # setup non-shuffle index to stride across flat keys properly
        try:
            datum = ImageMaskPair()  # create a datum for decoding serialized caffe_pb2 objects

            local_lmdb_txn = self.lmdb_txns[self.key_idx]

            # while the worker has not been told to terminate, loop infinitely
            while not termimation_flag:

                # poll termination queue for shutdown command
                try:
                    if self.terminateQ.get_nowait() is None:
                        termimation_flag = True
                        break
                except queue.Empty:
                    pass  # do nothing

                # build a single image selecting the labels using round robin through the shuffled order

                fn = self.__get_next_key()

                # extract the serialized image from the database
                value = local_lmdb_txn.get(fn)
                # convert from serialized representation
                datum.ParseFromString(value)

                # convert from string to numpy array
                I = np.fromstring(datum.image, dtype=datum.img_type)
                # reshape the numpy array using the dimensions recorded in the datum
                I = I.reshape((datum.img_height, datum.img_width, datum.channels))

                # convert from string to numpy array
                M = np.fromstring(datum.mask, dtype=datum.mask_type)
                # reshape the numpy array using the dimensions recorded in the datum
                M = M.reshape(datum.img_height, datum.img_width)

                if self.use_augmentation:
                    I = I.astype(np.float32)

                    # perform image data augmentation
                    I, M = augment_image(I, M,
                                         reflection_flag=self._reflection_flag,
                                         rotation_flag=self._rotation_flag,
                                         jitter_augmentation_severity=self._jitter_augmentation_severity,
                                         noise_augmentation_severity=self._noise_augmentation_severity,
                                         scale_augmentation_severity=self._scale_augmentation_severity,
                                         blur_augmentation_max_sigma=self._blur_max_sigma)

                # format the image into a tensor
                # reshape into tensor (CHW)
                I = I.transpose((2, 0, 1))
                I = I.astype(np.float32)
                I = zscore_normalize(I)

                M = M.astype(np.int32)
                # convert to a one-hot (HWC) representation
                h, w = M.shape
                M = M.reshape(-1)
                fM = np.zeros((len(M), self.nb_classes), dtype=np.int32)
                fM[np.arange(len(M)), M] = 1
                fM = fM.reshape((h, w, self.nb_classes))


                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((I, fM))

        except Exception as e:
            print('***************** Reader Error *****************')
            print(e)
            traceback.print_exc()
            print('***************** Reader Error *****************')
        finally:
            # when the worker terminates add a none to the output so the parent gets a shutdown confirmation from each worker
            self.outQ.put(None)

    def get_example(self):
        # get a ready to train batch from the output queue and pass to to the caller
        if self.outQ.qsize() < int(0.1*self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.5*self.maxOutQSize):
            print('Input Queue Starvation Over')
            self.queue_starvation = False
        return self.outQ.get()

    def generator(self):
        while True:
            batch = self.get_example()
            if batch is None:
                return
            yield batch

    def get_queue_size(self):
        return self.outQ.qsize()

    def get_tf_dataset(self):
        print('Creating Dataset')
        # wrap the input queues into a Dataset
        # this sets up the imagereader class as a Python generator
        # Images come in as HWC, and are converted into CHW for network
        image_shape = tf.TensorShape((self.image_size[2], self.image_size[0], self.image_size[1]))
        label_shape = tf.TensorShape((self.image_size[0], self.image_size[1], self.nb_classes))
        return tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))


