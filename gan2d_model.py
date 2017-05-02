from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, Reshape, Conv2D, UpSampling2D
from keras.layers import LeakyReLU, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras import objectives
import keras.backend as K
import numpy as np


def get_model(config):
    input_shape = tuple(config["input_shape"])

    generator = get_generator(nb_kers=config['generator']['nb_kers'],
                              nb_latent=config['generator']['nb_latent'],
                              wd=config['weight_decay'],
                              dr=config['dropout'])

    discriminator = get_discriminator(input_shape=input_shape,
                                      nb_kers=config['discriminator']['nb_kers'],
                                      wd=config['weight_decay'],
                                      dr=config['dropout'])

    gan = get_gan(nb_latent=config['generator']['nb_latent'],
                  generator=generator,
                  discriminator=discriminator)

    g_opt = Adam(lr=config['generator']['lr'], beta_1=config['generator']['beta_1'])
    d_opt = Adam(lr=config['discriminator']['lr'], beta_1=config['discriminator']['beta_1'])

    model = {'generator': generator,
             'discriminator': discriminator,
             'gan': gan}

    opt = {'generator': g_opt, 'discriminator': d_opt}

    return model, opt


def get_gan(nb_latent, generator, discriminator):
    gan_input = Input(shape=[nb_latent])
    l1 = generator(gan_input)
    
    discriminator.trainable = False
    l2 = discriminator(l1)
    return Model(gan_input, l2)


def get_generator(nb_kers, nb_latent, wd, dr):
    gen_input = Input(shape=[nb_latent], name='input')
    d1 = Dense(nb_kers*14*14)(gen_input)
    d1 = Dropout(dr)(d1)
    b1 = BatchNormalization()(d1)
    a1 = Activation('relu')(b1)
    r1 = Reshape((nb_kers, 14, 14))(a1)
    u1 = UpSampling2D(size=(2, 2))(r1)

    c1 = Conv2D(nb_kers//2, 3, 3, border_mode='same')(u1)
    b2 = BatchNormalization()(c1)
    a2 = Activation('relu')(b2)
    a2 = Dropout(dr)(a2)

    c2 = Conv2D(nb_kers//4, 3, 3, border_mode='same')(a2)
    b3 = BatchNormalization()(c2)
    a3 = Activation('relu')(b3)
    c4 = Conv2D(1, 1, 1, border_mode='same', activation='tanh')(a3)

    return Model(gen_input, c4)


def get_discriminator(input_shape, nb_kers, dr, wd):
    discr_input = Input(shape=input_shape, name='input')

    c1 = Conv2D(nb_kers*2, 5, 5, border_mode='same', activation='relu')(discr_input)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    a1 = Dropout(dr)(p1)

    c2 = Conv2D(nb_kers*4, 5, 5, border_mode='same', activation='relu')(a1)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    a2 = Dropout(dr)(p2)

    d4 = Flatten()(a2)
    d4 = Dense(1024, activation='relu')(d4)
    a4 = Dropout(dr)(d4)

    d5 = Dense(1, activation='sigmoid')(a4)

    return Model(discr_input, d5)


