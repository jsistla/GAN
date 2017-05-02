__author__ = 'tnair'
import os
import json
import argparse
import numpy as np
import gan2d_model
from keras.datasets import mnist
from timeit import default_timer as timer
from keras.optimizers import Adam

_MODEL_TYPES = {"gan_2D": gan2d_model}


def _get_cfg():
    parser = argparse.ArgumentParser(description="Main handler for training GAN",
                                     usage="./scripts/training/train.sh -j training_config.json")

    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    args = parser.parse_args()
    with open(args.json, 'r') as f:
        cfg = json.loads(f.read())

    return cfg


def main(cfg):

    expt_cfg = cfg['experiment']
    expt_name = expt_cfg['name']
    outdir = expt_cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, expt_name), exist_ok=True)

    model_cfg = cfg['model']
    model_type = _MODEL_TYPES[model_cfg['type']]

    model, optimizer = model_type.get_model(model_cfg)

    generator = model['generator']
    gan = model['gan']
    discriminator = model['discriminator']

    discriminator.trainable = False
    generator.compile(loss='binary_crossentropy', optimizer=optimizer['generator'])
    gan.compile(loss='binary_crossentropy', optimizer=optimizer['generator'])

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer['discriminator'], metrics=['accuracy'])

    (x_train_real, y_train_real), (x_test, y_test) = mnist.load_data()
    x_train_real = x_train_real.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train_real /= 255.0
    x_test /= 255.0

    x_train_real = np.expand_dims(x_train_real, axis=1)
    x_train_real = np.reshape(x_train_real, (len(x_train_real), -1))
    np.random.shuffle(x_train_real)

    training_start = timer()
    for e in range(expt_cfg['nb_epochs']):
        z_gen_concat = []
        d_loss_concat = []

        if e >= 100:
            discriminator.optimizer = Adam(lr=1e-5, beta_1=0.5)
            gan.optimizer = Adam(lr=9e-5, beta_1=0.5)

        print("{:.2f}s Epoch {}".format((timer() - training_start), e), end='')
        discriminator.trainable = True
        epoch_start = timer()
        for k in range(expt_cfg['nb_discriminator_steps_per_epoch']):
            # sample mini-batch of m noise samples from noise prior
            z = np.asarray([np.random.uniform(-1, 1, model_cfg['generator']['nb_latent'])
                            for _ in range(expt_cfg['nb_z_samples'])])
            z_gen_concat.append(z)

            # sample minibatch of m examples X[1:m] from data generator
            x_gen = generator.predict(z)
            x_gen = np.reshape(x_gen, (len(x_gen), -1))

            # update discriminator in two batches: one for the real, one for the fake
            discriminator.trainable = True
            d_loss0 = discriminator.train_on_batch(x_train_real[:expt_cfg['nb_z_samples'], :].reshape(-1,1,28,28), np.ones(shape=len(x_gen)))
            d_loss1 = discriminator.train_on_batch(x_gen.reshape(-1, 1, 28, 28), np.zeros(shape=len(x_gen)))
            d_loss_concat.append(d_loss0)
            d_loss_concat.append(d_loss1)
            discriminator.trainable = False

        # sample minibatch of m noise samples from noise prior
        # actually, I can just keep track of the images I've already generated and use those
        z_gen_concat = np.asarray(z_gen_concat).reshape(-1, model_cfg['generator']['nb_latent'])

        y_gen = np.ones((len(z_gen_concat), 1))

        # update generator by descending its stochastic gradient
        discriminator.trainable = False
        g_loss = gan.train_on_batch(z_gen_concat, y_gen)

        # check how the discriminator is doing
        x_gen = generator.predict(z_gen_concat)
        x_gen = np.reshape(x_gen, (len(x_gen), -1))
        X = np.concatenate((x_train_real[:expt_cfg['nb_z_samples'], :], x_gen), axis=0)
        X = X.reshape(2 * expt_cfg['nb_z_samples'], 1, 28, 28)
        y = np.concatenate((np.ones(shape=len(x_gen)), np.zeros(shape=(len(x_gen)))), axis=0)[:, np.newaxis]
        d_loss = discriminator.evaluate(X, y, batch_size=128, verbose=0)

        print('     {:.2f}s     GAN loss: {:.6e}    Discriminator loss: {:.6e}      accuracy: {}'.format(
            (timer()-epoch_start)/60, float(g_loss), float(d_loss[0]), float(d_loss[1])))

        generator.save_weights(os.path.join(outdir, expt_name,
                                            'weights_epoch{}_gloss{:.4e}.hdf5'.format(e, float(g_loss))))

    generator.save_weights(os.path.join(outdir, expt_name, 'weights_epoch{}_gloss{:.4e}.hdf5'.format(e, float(g_loss))))


if __name__ == "__main__":
    main(_get_cfg())
