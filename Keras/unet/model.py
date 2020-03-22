from keras.backend import int_shape
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Input, Activation, Concatenate
from keras import initializers


def unet(filter_root, depth, n_class=2, input_size=(256, 256, 1), activation='relu', batch_norm=True, final_activation='sigmoid'):
    """
    Build UNet model with ResBlock.

    Args:
        filter_root (int): Number of filters to start with in first convolution.
        depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model. 
                    Filter root and image size should be multiple of 2^depth.
        n_class (int, optional): How many classes in the output layer. Defaults to 2.
        input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
        activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
        batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
        final_activation (str, optional): activation for output layer. Defaults to 'softmax'.

    Returns:
        obj: keras model object
    """
    inputs = Input(input_size)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}

    # Down sampling
    for i in range(depth):
        out_channel = 2**i * filter_root

        # First Conv2D Block with Conv2D, BN and activation
        conv1 = Conv2D(out_channel, kernel_size=3, padding='same', activation="relu", name="Conv2D{}_1".format(i))(x)

        # Second Conv2D block with Conv2D and BN only
        conv2 = Conv2D(out_channel, kernel_size=3, padding='same', activation="relu", name="Conv2D{}_2".format(i))(conv1)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = conv2
            x = MaxPooling2D(padding='same', name="MaxPooling2D{}_1".format(i))(conv2)
        else:
            x = conv2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        out_channel = 2**(i) * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up1 = UpSampling2D(name="UpSampling2D{}_1".format(i))(x)
        up_conv1 = Conv2D(out_channel, 2, activation='relu', padding='same', name="upsamplingConv{}_1".format(i))(up1)

        #  Concatenate.
        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])

        #  Convolutions
        up_conv1 = Conv2D(out_channel, 3, padding='same', activation="relu", name="upConv{}_1".format(i))(up_conc)
        up_conv2 = Conv2D(out_channel, 3, padding='same', activation="relu", name="upConv{}_2".format(i))(up_conv1)

        x = up_conv2

    # Final convolution
    output = Conv2D(n_class, 1, padding='same', activation="sigmoid", name='output')(x)

    return Model(inputs, outputs=output, name='Res-UNet')
