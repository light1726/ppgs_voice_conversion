import os
import argparse
import tensorflow as tf

from hparams import hparams
from models import get_biGRU_model
from data_reader import train_generator, dev_generator, process


def main():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    args = parser.parse_args()

    # get data
    train_set = tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, tf.int32, tf.float32),
        output_shapes=([None, hparams['input_dim']], [],
                       [None, hparams['output_dim']]))
    train_set = train_set.padded_batch(hparams['batch_size'],
                                       ([None, hparams['input_dim']], [],
                                        [None, hparams['output_dim']]))
    train_set = train_set.map(process)

    dev_set = tf.data.Dataset.from_generator(
        dev_generator,
        output_types=(tf.float32, tf.int32, tf.float32),
        output_shapes=([None, hparams['batch_size']], [],
                       [None, hparams['output_dim']]))
    dev_set = dev_set.padded_batch(hparams['batch_size'],
                                   ([None, hparams['input_dim']], [],
                                    [None, hparams['output_dim']]))
    dev_set = dev_set.map(process)
    # build model
    model = get_biGRU_model(in_dim=hparams['input_dim'],
                            initial_hidden=hparams['input_dense_hidden'],
                            drop_rate=hparams['drop_rate'],
                            gru_hidden=hparams['gru_hidden'],
                            out_dim=hparams['output_dim'])
    call_backs = [tf.keras.callbacks.TensorBoard(args.log_dir),
                  tf.keras.callbacks.ModelCheckpoint(
                      os.path.join(args.model_dir, 'weights.{epoch:02d}.hdf5'))]
    model.compile(optimizer=tf.keras.optimizers.Adam(hparams['learning_rate']),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.losses.MeanSquaredError()])
    model.summary()
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
    model.fit(train_set, epochs=hparams['epochs'],
              callbacks=call_backs,
              validation_data=dev_set)


if __name__ == '__main__':
    main()
