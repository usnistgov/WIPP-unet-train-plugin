# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
if sys.version_info[0] < 3:
    raise Exception('Python3 required')

import os
# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi

import argparse
import numpy as np

import tensorflow as tf
tf_version = tf.__version__.split('.')
if int(tf_version[0]) != 2:
    raise Exception('Tensorflow 2.x.x required')

import unet_model
import imagereader
import build_lmdb
import tempfile

EARLY_STOPPING_COUNT = 10
CONVERGENCE_TOLERANCE = 1e-4
READER_COUNT = 1 # 1 per GPU, both the reader count and batch size will be scaled based on the number of GPUs


def train_model(output_folder, scratch_dir, batch_size, train_lmdb_filepath, test_lmdb_filepath, number_classes, balance_classes, learning_rate, test_every_n_steps, use_augmentation, augmentation_reflection, augmentation_rotation, augmentation_jitter, augmentation_noise, augmentation_scale, augmentation_blur_max_sigma):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    training_checkpoint_filepath = None

    # use all available devices
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
        # scale the number of I/O readers based on the GPU count
        reader_count = READER_COUNT * mirrored_strategy.num_replicas_in_sync

        print('Setting up test image reader')
        test_reader = imagereader.ImageReader(test_lmdb_filepath, use_augmentation=False, shuffle=False, num_workers=reader_count, balance_classes=False, number_classes=number_classes)
        print('Test Reader has {} images'.format(test_reader.get_image_count()))

        print('Setting up training image reader')
        train_reader = imagereader.ImageReader(train_lmdb_filepath, use_augmentation=use_augmentation, shuffle=True, num_workers=reader_count, balance_classes=balance_classes, number_classes=number_classes, augmentation_reflection=augmentation_reflection, augmentation_rotation=augmentation_rotation, augmentation_jitter=augmentation_jitter, augmentation_noise=augmentation_noise, augmentation_scale=augmentation_scale, augmentation_blur_max_sigma=augmentation_blur_max_sigma)
        print('Train Reader has {} images'.format(train_reader.get_image_count()))

        try:  # if any errors happen we want to catch them and shut down the multiprocess readers
            print('Starting Readers')
            train_reader.startup()
            test_reader.startup()

            train_dataset = train_reader.get_tf_dataset()
            train_dataset = train_dataset.batch(global_batch_size).prefetch(reader_count)
            train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

            test_dataset = test_reader.get_tf_dataset()
            test_dataset = test_dataset.batch(global_batch_size).prefetch(reader_count)
            test_dataset = mirrored_strategy.experimental_distribute_dataset(test_dataset)

            print('Creating model')
            model = unet_model.UNet(number_classes, global_batch_size, train_reader.get_image_size(), learning_rate)

            checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())

            # print the model summary to file
            with open(os.path.join(output_folder, 'model.txt'), 'w') as summary_fh:
                print_fn = lambda x: print(x, file=summary_fh)
                model.get_keras_model().summary(print_fn=print_fn)

            # train_epoch_size = train_reader.get_image_count()/batch_size
            train_epoch_size = test_every_n_steps
            test_epoch_size = test_reader.get_image_count() / batch_size

            test_loss = list()

            # Prepare the metrics.
            train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            train_acc_metric = tf.keras.metrics.CategoricalAccuracy('train_accuracy')
            test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
            test_acc_metric = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

            train_log_dir = os.path.join(output_folder, 'tensorboard','train')
            if not os.path.exists(train_log_dir):
                os.makedirs(train_log_dir)
            test_log_dir = os.path.join(output_folder, 'tensorboard','test')
            if not os.path.exists(test_log_dir):
                os.makedirs(test_log_dir)

            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

            epoch = 0
            print('Running Network')
            while True:  # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))

                if epoch == 0:
                    cur_train_epoch_size = min(1000, train_epoch_size)
                    print('Performing Adam Optimizer learning rate warmup for {} steps'.format(cur_train_epoch_size))
                    model.set_learning_rate(learning_rate / 10)
                else:
                    cur_train_epoch_size = train_epoch_size
                    model.set_learning_rate(learning_rate)

                # Iterate over the batches of the train dataset.
                for step, (batch_images, batch_labels) in enumerate(train_dataset):
                    if step > train_epoch_size:
                        break

                    inputs = (batch_images, batch_labels, train_loss_metric, train_acc_metric)
                    model.dist_train_step(mirrored_strategy, inputs)

                    print('Train Epoch {}: Batch {}/{}: Loss {} Accuracy = {}'.format(epoch, step, train_epoch_size, train_loss_metric.result(), train_acc_metric.result()))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss_metric.result(), step=int(epoch * train_epoch_size + step))
                        tf.summary.scalar('accuracy', train_acc_metric.result(), step=int(epoch * train_epoch_size + step))
                    train_loss_metric.reset_states()
                    train_acc_metric.reset_states()

                # Iterate over the batches of the test dataset.
                epoch_test_loss = list()
                for step, (batch_images, batch_labels) in enumerate(test_dataset):
                    if step > test_epoch_size:
                        break

                    inputs = (batch_images, batch_labels, test_loss_metric, test_acc_metric)
                    loss_value = model.dist_test_step(mirrored_strategy, inputs)

                    epoch_test_loss.append(loss_value.numpy())
                    # print('Test Epoch {}: Batch {}/{}: Loss {}'.format(epoch, step, test_epoch_size, loss_value))
                test_loss.append(np.mean(epoch_test_loss))

                print('Test Epoch: {}: Loss = {} Accuracy = {}'.format(epoch, test_loss_metric.result(), test_acc_metric.result()))
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_loss_metric.result(), step=int((epoch+1) * train_epoch_size))
                    tf.summary.scalar('accuracy', test_acc_metric.result(), step=int((epoch+1) * train_epoch_size))
                test_loss_metric.reset_states()
                test_acc_metric.reset_states()

                with open(os.path.join(output_folder, 'test_loss.csv'), 'w') as csvfile:
                    for i in range(len(test_loss)):
                        csvfile.write(str(test_loss[i]))
                        csvfile.write('\n')

                # determine if to record a new checkpoint based on best test loss
                if (len(test_loss) - 1) == np.argmin(test_loss):
                    # save tf checkpoint
                    print('Test loss improved: {}, saving checkpoint'.format(np.min(test_loss)))
                    training_checkpoint_filepath = checkpoint.write(os.path.join(scratch_dir, "ckpt"))

                # determine early stopping
                print('Best Current Epoch Selection:')
                print('Test Loss:')
                print(test_loss)
                min_test_loss = np.min(test_loss)
                error_from_best = np.abs(test_loss - min_test_loss)
                error_from_best[error_from_best < CONVERGENCE_TOLERANCE] = 0
                best_epoch = np.where(error_from_best == 0)[0][0] # unpack numpy array, select first time since that value has happened
                print('Best epoch: {}'.format(best_epoch))

                if len(test_loss) - best_epoch > EARLY_STOPPING_COUNT:
                    break  # break the epoch loop
                epoch = epoch + 1

        finally: # if any errors happened during training, shut down the disk readers
            print('Shutting down train_reader')
            train_reader.shutdown()
            print('Shutting down test_reader')
            test_reader.shutdown()

    if training_checkpoint_filepath is not None:
        # restore the checkpoint and generate a saved model
        model = unet_model.UNet(number_classes, global_batch_size, train_reader.get_image_size(), learning_rate)
        checkpoint = tf.train.Checkpoint(optimizer=model.get_optimizer(), model=model.get_keras_model())
        checkpoint.restore(training_checkpoint_filepath)
        tf.saved_model.save(model.get_keras_model(), output_folder)


def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

    # lmdb parameters
    parser.add_argument('--imageDir', dest='image_dir', type=str, help='filepath to the directory containing the images', required=True)
    parser.add_argument('--maskDir', dest='mask_dir', type=str, help='filepath to the directory containing the masks', required=True)
    parser.add_argument('--useTiling', dest='use_tiling', type=str, help='whether to use tiling when training [YES, NO]', default="NO")
    parser.add_argument('--tileSize', dest='tile_size', type=int, default=256)
    parser.add_argument('--trainFraction', dest='train_fraction', type=float, help='what fraction of the dataset to use for training (0.0, 1.0)', default=0.8)

    # Training parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, help='training batch size', default=4)
    parser.add_argument('--numberClasses', dest='number_classes', type=int, default=2)
    parser.add_argument('--learningRate', dest='learning_rate', type=float, default=1e-4)
    parser.add_argument('--outputDir', dest='output_dir', type=str, help='Folder where outputs will be saved (Required)', required=True)
    parser.add_argument('--testEveryNSteps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=1000)
    parser.add_argument('--balanceClasses', dest='balance_classes', type=str, help='whether to balance classes [YES, NO]', default="NO")

    # Augmentation parameters
    parser.add_argument('--useAugmentation', dest='use_augmentation', type=str, help='whether to use data augmentation [YES, NO]', default="NO")
    parser.add_argument('--augmentationReflection', dest='augmentation_reflection', type=str, help='whether to use reflection data augmentation [YES, NO]', default="YES")
    parser.add_argument('--augmentationRotation', dest='augmentation_rotation', type=str, help='whether to use roation data augmentation [YES, NO]', default="YES")
    parser.add_argument('--augmentationJitter', dest='augmentation_jitter', type=float, help='jitter data augmentation severity [0 = none, 1 = 100% image size], default = 0.1 (10% image size)', default=0.1)
    parser.add_argument('--augmentationNoise', dest='augmentation_noise', type=float, help='noise data augmentation severity as a percentage of the image dynamic range [0 = none, 1 = 100%], default = 0.02 (2% dynamic range)', default=0.02)
    parser.add_argument('--augmentationScale', dest='augmentation_scale', type=float, help='scale data augmentation severity as a percentage of the image size [0 = none, 1 = 100%], default = 0.1 (10% max change in image size)', default=0.1)
    parser.add_argument('--augmentationBlurMaxSigma', dest='augmentation_blur_max_sigma', type=float, help='maximum sigma to use in a gaussian blurring kernel. Blur kernel is selected as rand(0, max)', default=2)


    print('Arguments:')
    args = parser.parse_args()

    # lmdb parameters
    use_tiling = args.use_tiling
    use_tiling = use_tiling.upper() == "YES"

    tile_size = args.tile_size
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    train_fraction = args.train_fraction

    print('use_tiling = {}'.format(use_tiling))
    print('tile_size = {}'.format(tile_size))
    print('image_dir = {}'.format(image_dir))
    print('mask_dir = {}'.format(mask_dir))
    print('train_fraction = []'.format(train_fraction))

    # Training parameters
    batch_size = args.batch_size
    output_dir = args.output_dir
    number_classes = args.number_classes

    learning_rate = args.learning_rate
    test_every_n_steps = args.test_every_n_steps
    balance_classes = args.balance_classes
    balance_classes = balance_classes.upper() == "YES"

    print('batch_size = {}'.format(batch_size))
    print('number_classes = {}'.format(number_classes))
    print('learning_rate = {}'.format(learning_rate))
    print('test_every_n_steps = {}'.format(test_every_n_steps))
    print('balance_classes = {}'.format(balance_classes))
    print('output_dir = {}'.format(output_dir))

    # Augmentation parameters
    use_augmentation = args.use_augmentation
    use_augmentation = use_augmentation.upper() == "YES"
    print('use_augmentation = {}'.format(use_augmentation))
    if use_augmentation:
        augmentation_reflection = args.augmentation_reflection
        augmentation_reflection = augmentation_reflection.upper() == "YES"
        augmentation_rotation = args.augmentation_rotation
        augmentation_rotation = augmentation_rotation.upper() == "YES"
        augmentation_jitter = args.augmentation_jitter
        augmentation_noise = args.augmentation_noise
        augmentation_scale = args.augmentation_scale
        augmentation_blur_max_sigma = args.augmentation_blur_max_sigma

        print('augmentation_reflection = {}'.format(augmentation_reflection))
        print('augmentation_rotation = {}'.format(augmentation_rotation))
        print('augmentation_jitter = {}'.format(augmentation_jitter))
        print('augmentation_noise = {}'.format(augmentation_noise))
        print('augmentation_scale = {}'.format(augmentation_scale))
        print('augmentation_blur_max_sigma = {}'.format(augmentation_blur_max_sigma))

    else:
        augmentation_reflection = 0
        augmentation_rotation = 0
        augmentation_jitter = 0
        augmentation_noise = 0
        augmentation_scale= 0
        augmentation_blur_max_sigma = 0

    dataset_name = 'unet'
    image_format = 'tif'

    # create the scratch directory so that it gets self cleaned up after with block
    with tempfile.TemporaryDirectory() as scratch_dir:
        if not use_tiling:
            tile_size = 0
        train_database_name, test_database_name = build_lmdb.build_database(image_dir, mask_dir, scratch_dir, dataset_name, train_fraction, image_format, tile_size)
        train_lmdb_filepath = os.path.join(scratch_dir, train_database_name)
        test_lmdb_filepath = os.path.join(scratch_dir, test_database_name)

        train_model(output_dir, scratch_dir, batch_size, train_lmdb_filepath, test_lmdb_filepath, number_classes, balance_classes, learning_rate, test_every_n_steps, use_augmentation, augmentation_reflection, augmentation_rotation, augmentation_jitter, augmentation_noise, augmentation_scale, augmentation_blur_max_sigma)




if __name__ == "__main__":
    main()
