# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise RuntimeError('Python3 required')

import skimage.io
import numpy as np
import os
import skimage
import skimage.transform
from isg_ai_pb2 import ImageMaskPair
import shutil
import lmdb
import random
import argparse
import unet_model


def read_image(fp):
    img = skimage.io.imread(fp)
    return img


def write_img_to_db(txn, img, msk, key_str):
    if type(img) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if type(msk) is not np.ndarray:
        raise Exception("Img must be numpy array to store into db")
    if len(img.shape) > 3:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")
    if len(img.shape) < 2:
        raise Exception("Img must be 2D or 3D [HW, or HWC] format")

    if len(img.shape) == 2:
        # make a 3D array
        img = img.reshape((img.shape[0], img.shape[1], 1))

    # get the list of labels in the image
    labels = np.unique(msk)

    datum = ImageMaskPair()
    datum.channels = img.shape[2]
    datum.img_height = img.shape[0]
    datum.img_width = img.shape[1]

    datum.img_type = img.dtype.str
    datum.mask_type = msk.dtype.str

    datum.image = img.tobytes()
    datum.mask = msk.tobytes()

    datum.labels = labels.tobytes()

    txn.put(key_str.encode('ascii'), datum.SerializeToString())
    return


def enforce_size_multiple(img):
    h = img.shape[0]
    w = img.shape[1]

    # this function crops the input image down slightly to be a size multiple of 16

    factor = unet_model.UNet.SIZE_FACTOR
    tgt_h = int(np.floor(h / factor) * factor)
    tgt_w = int(np.floor(w / factor) * factor)

    dh = h - tgt_h
    dw = w - tgt_w

    img = img[int(dh/2):, int(dw/2):]
    img = img[0:tgt_h, 0:tgt_w]

    return img


def process_slide_tiling(img, msk, tile_size):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]

    img_list = []
    msk_list = []

    delta = int(tile_size - int(0.1 * tile_size))

    for x_st in range(0, width, delta):
        for y_st in range(0, height, delta):
            x_end = x_st + tile_size
            y_end = y_st + tile_size
            if x_st < 0 or y_st < 0:
                # should never happen, but stranger things and all...
                continue
            if x_end > width:
                # slide box to fit within image
                dx = width - x_end
                x_st = x_st + dx
                x_end = x_end + dx
            if y_end > height:
                # slide box to fit within image
                dy = height - y_end
                y_st = y_st + dy
                y_end = y_end + dy

            # handle if the image is smaller than the tile size
            x_st = max(0, x_st)
            y_st = max(0, y_st)

            # crop out the tile
            img_pixels = img[y_st:y_end, x_st:x_end]
            msk_pixels = msk[y_st:y_end, x_st:x_end]

            img_list.append(img_pixels)
            msk_list.append(msk_pixels)

    return (img_list, msk_list)


def generate_database_tiling(img_list, database_name, image_filepath, mask_filepath, output_folder, tile_size):
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    with open(os.path.join(output_image_lmdb_file, 'img_filenames.csv'), 'w') as csvfile:
        for fn in img_list:
            csvfile.write(fn + '\n')

    for i in range(len(img_list)):
        print('  {}/{}'.format(i, len(img_list)))
        txn_nb = 0
        img_file_name = img_list[i]
        block_key = img_file_name.replace('.tif','')

        img = read_image(os.path.join(image_filepath, img_file_name))
        msk = read_image(os.path.join(mask_filepath, img_file_name))
        msk = msk.astype(np.uint8)
        assert img.shape[0] == msk.shape[0], 'Image and Mask must be the same Height'
        assert img.shape[1] == msk.shape[1], 'Image and Mask must be the same Width'

        # convert the image mask pair into tiles
        img_tile_list, msk_tile_list = process_slide_tiling(img, msk, tile_size)

        for k in range(len(img_tile_list)):
            img = img_tile_list[k]
            msk = msk_tile_list[k]
            key_str = '{}_{:08d}'.format(block_key, txn_nb)
            txn_nb += 1
            write_img_to_db(image_txn, img, msk, key_str)

            if txn_nb % 1000 == 0:
                image_txn.commit()
                image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()


def generate_database(img_list, database_name, image_filepath, mask_filepath, output_folder):
    output_image_lmdb_file = os.path.join(output_folder, database_name)

    if os.path.exists(output_image_lmdb_file):
        print('Deleting existing database')
        shutil.rmtree(output_image_lmdb_file)

    image_env = lmdb.open(output_image_lmdb_file, map_size=int(5e12))
    image_txn = image_env.begin(write=True)

    with open(os.path.join(output_image_lmdb_file, 'img_filenames.csv'), 'w') as csvfile:
        for fn in img_list:
            csvfile.write(fn + '\n')

    largest_height = 0
    largest_width = 0
    for i in range(len(img_list)):
        txn_nb = 0
        img_file_name = img_list[i]
        block_key = img_file_name.replace('.tif','')

        img = read_image(os.path.join(image_filepath, img_file_name))
        msk = read_image(os.path.join(mask_filepath, img_file_name))
        msk = msk.astype(np.uint8)
        assert img.shape[0] == msk.shape[0], 'Image and Mask must be the same Height'
        assert img.shape[1] == msk.shape[1], 'Image and Mask must be the same Width'
        largest_height = max(largest_height, img.shape[0])
        largest_width = max(largest_width, img.shape[1])

        img = enforce_size_multiple(img)
        msk = enforce_size_multiple(msk)

        key_str = '{}_{:08d}'.format(block_key, txn_nb)
        txn_nb += 1
        write_img_to_db(image_txn, img, msk, key_str)

        if txn_nb % 1000 == 0:
            image_txn.commit()
            image_txn = image_env.begin(write=True)

    image_txn.commit()
    image_env.close()
    return largest_height, largest_width


def build_database(image_folder, mask_folder, output_folder, dataset_name, train_fraction, image_format, tile_size):

    if image_format.startswith('.'):
        # remove leading period
        image_format = image_format[1:]

    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # find the image files for which annotations exist
    img_files = [f for f in os.listdir(mask_folder) if f.endswith('.{}'.format(image_format))]

    print('len(img_files)', len(img_files))
    if len(img_files) == 0:
        msg = "Found no image files with the provided image format {}".format(image_format)
        raise RuntimeError(msg)

    # in place shuffle
    random.shuffle(img_files)

    idx = int(train_fraction * len(img_files))
    train_img_files = img_files[0:idx]
    test_img_files = img_files[idx:]

    print('INFO: tile_size in build_databse:', tile_size)
    if len(train_img_files) == 0:
        raise RuntimeError("Train dataset contains 0 images.")
    if len(test_img_files) == 0:
        raise RuntimeError("Test dataset contains 0 images.")

    if tile_size == 0:
        print('building train database')
        train_database_name = 'train-{}.lmdb'.format(dataset_name)
        largest_height, largest_width = generate_database(train_img_files, train_database_name, image_folder, mask_folder, output_folder)

        print('building test database')
        test_database_name = 'test-{}.lmdb'.format(dataset_name)
        largest_height2, largest_width2 = generate_database(test_img_files, test_database_name, image_folder, mask_folder, output_folder)
        largest_height = max(largest_height, largest_height2)
        largest_width = max(largest_width, largest_width2)

    else:
        print('INFO: check tile_size % size factor:', (tile_size % unet_model.UNet.SIZE_FACTOR) )

        assert tile_size % unet_model.UNet.SIZE_FACTOR == 0, 'UNet requires tiles with shapes that are multiples of 16'
        print('building train database')
        train_database_name = 'train-{}.lmdb'.format(dataset_name)
        generate_database_tiling(train_img_files, train_database_name, image_folder, mask_folder, output_folder, tile_size)

        print('building test database')
        test_database_name = 'test-{}.lmdb'.format(dataset_name)
        generate_database_tiling(test_img_files, test_database_name, image_folder, mask_folder, output_folder, tile_size)
        largest_height = tile_size
        largest_width = tile_size

    return train_database_name, test_database_name, [largest_height, largest_width]


def main():
    # Define the inputs
    # ****************************************************

    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='build_lmdb', description='Script which converts two folders of images and masks into a pair of lmdb databases for training.')

    parser.add_argument('--image_folder', dest='image_folder', type=str, help='filepath to the folder containing the images', required=True)
    parser.add_argument('--mask_folder', dest='mask_folder', type=str, help='filepath to the folder containing the masks', required=True)
    parser.add_argument('--output_folder', dest='output_folder', type=str, help='filepath to the folder where the outputs will be placed', required=True)
    parser.add_argument('--dataset_name', dest='dataset_name', type=str, help='name of the dataset to be used in creating the lmdb files', required=True)
    parser.add_argument('--train_fraction', dest='train_fraction', type=float, help='what fraction of the dataset to use for training (0.0, 1.0)', default=0.8)
    parser.add_argument('--image_format', dest='image_format', type=str, help='format (extension) of the input images. E.g {tif, jpg, png)', default='tif')
    parser.add_argument('--tile_size', dest='tile_size', type=int, help='The size of the tiles to crop out of the source images, striding across all available pixels in the source images, use tile_size=0 to turn off tiling', default=0)

    args = parser.parse_args()
    image_folder = args.image_folder
    mask_folder = args.mask_folder
    output_folder = args.output_folder
    dataset_name = args.dataset_name
    train_fraction = args.train_fraction
    image_format = args.image_format
    tile_size = args.tile_size

    train_database_name, test_database_name, largest_image_shape = build_database(image_folder, mask_folder, output_folder, dataset_name, train_fraction, image_format, tile_size)

if __name__ == "__main__":
    main()


