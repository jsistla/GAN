# GAN
A clean, and commented GAN implementation for MNIST.
Usage:

> python gan2d_train.py -j config.json

As you can tell from the last frame of the gif below, it's definitely not perfect. That said, the GAN does not collapse into reproducing the same image, the generator and disccriminator losses are quite stable, and the generated images do resemble digits.

Takeaways from training:
1. Use tanh as the output activation on the generator
2. Train the discriminator with batches containing a single class (ie. only real or only fake images)
3. Draw from a distribution on [-1,1] instead of [0,1]
4. Reduce the default Keras Adam momentum rate.
5. Reduce the learning rate during training

Instead of trying to perfectly generate MNIST, I'm going to take what I've learned at move to more interesting datasets!


Training GIF:

![Alt text](https://github.com/tanyanair/GAN/blob/master/readme_objs/mnist_training.gif?raw=True "MNIST Training")
