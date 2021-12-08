import tensorflow as tf
from tensorflow import keras


def unet_Original(input_size=(32, 32, 5)):
    filtersFirstLayer = 32
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(filtersFirstLayer, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(filtersFirstLayer, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(filtersFirstLayer * 2, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        pool1)
    conv2 = tf.keras.layers.Conv2D(filtersFirstLayer * 2, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(filtersFirstLayer * 4, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        pool2)
    conv3 = tf.keras.layers.Conv2D(filtersFirstLayer * 4, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(filtersFirstLayer * 8, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        pool3)
    conv4 = tf.keras.layers.Conv2D(filtersFirstLayer * 8, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(filtersFirstLayer * 16, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        pool4)
    conv5 = tf.keras.layers.Conv2D(filtersFirstLayer * 16, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv5)

    up6 = tf.keras.layers.Conv2D(filtersFirstLayer * 8, 2, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(filtersFirstLayer * 8, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        merge6)
    conv6 = tf.keras.layers.Conv2D(filtersFirstLayer * 8, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv6)

    up7 =  tf.keras.layers.Conv2D(filtersFirstLayer * 4, 2, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 =  tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 =  tf.keras.layers.Conv2D(filtersFirstLayer * 4, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        merge7)
    conv7 =  tf.keras.layers.Conv2D(filtersFirstLayer * 4, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv7)

    up8 =  tf.keras.layers.Conv2D(filtersFirstLayer * 2, 2, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 =  tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 =  tf.keras.layers.Conv2D(filtersFirstLayer * 2, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        merge8)
    conv8 =  tf.keras.layers.Conv2D(filtersFirstLayer * 2, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        conv8)

    up9 =  tf.keras.layers.Conv2D(filtersFirstLayer, 2, activation='relu', padding='same', kernel_initializer='glorot_normal')(
        tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(filtersFirstLayer, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(filtersFirstLayer, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(conv9)
    conv9 =  tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='glorot_normal')(conv9)
    conv10 =  tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs, conv10)

    return model


def conv_block(input_tensor, n_filters, kernel_size=3):
    """
    Adiciona 2 blocos convolucionais ativados com a função ReLU
    :param input_tensor: (tuple) dimensão do tensor de input.
    :param n_filters: (int) quantidade de filtros.
    :param kernel_size: tamanho do kernel
    :return: 2 camadas de convolução ativadas com a função ReLU.
    """

    inputs = input_tensor

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                                   kernel_initializer="he_normal", padding="same")(inputs)
        x = tf.keras.layers.Activation("relu")(x)

        return x


def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    """
    Criar o bloco encoder com duas camadas de convolução, uma camada de maxpooling e uma de dropout.

    :param inputs: (tf.tensor) tensor de entrada.
    :param n_filters: (int) quantidade de filtros. default = 64
    :param pool_size:  (tuple) dimensão do pooling. default = (2,2)
    :param dropout:  (float) probabilidade do dropout. default = 0.3
    :return:
    conv (tf.tensor) resultado da convolução;
    p (tf.tensor) resultado da operação de pooling e dropout.
    """
    # duas camadas de convolução
    conv = conv_block(inputs, n_filters=n_filters)
    # camada de max pooling
    p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv)
    # camada de dropout
    p = tf.keras.layers.Dropout(dropout)(p)

    return conv, p



def encoder(inputs):
    """

    :param inputs: (tf.tensor) tensor de etrada.
    :return:
    p4 (tf.tensor) output do maxpooling do último bloco de encoder.
    (f1,f2,f3,f4) (tf.tensor) output the cada bloco de encoder.
    """

    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2,2),dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2, 2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2, 2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2, 2), dropout=0.3)

    return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
    """
    :param inputs: (tf.tensor) tensor de entrada.
    :return: (tf.tensor) saída do bottleneck.
    """

    bottleneck = conv_block(inputs, n_filters=1024)

    return bottleneck

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    """
    Bloco decoder da rede unet.

    :param inputs: (tf.tensor) tensor de entrada.
    :param conv_output: (tf.tensor) tensor obtido no caminho encoder que será concatenado.
    :param n_filters: (int) número de filtros.
    :param kernel_size: (int) dimensão do kernel.
    :param strides: (int) deslocamento lateral dos filtros.
    :param dropout: (float)  probabilidade de desativação dos filtros.
    :return:
    concat (tf.tensor) resultado do bloco de convolução
    """

    up = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding="same")(inputs)
    concat = tf.keras.layers.Concatenate()([up, conv_output])
    concat = tf.keras.layers.Dropout(dropout)(concat)
    concat = conv_block(concat,n_filters,kernel_size=3)

    return concat



def decoder(inputs, convs, output_channels, activation="sigmoid"):
    """
    Etapa decoder unet

    :param inputs: (tf.tensor) tensor de entrada
    :param convs: (tuple(tf.tensor)) tupla contendo os tensores da etapa encoder
    :param output_channels: (int) quantidade de canais de saida
    :param activation: (string) fução de ativação de saída. default = sigmoid
    :return:
    """

    f1, f2, f3, f4 = convs

    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=3, strides=2, dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=3, strides=2, dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=3, strides=2, dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=3, strides=2, dropout=0.3)

    outputs = tf.keras.layers.Conv2D(output_channels, (1,1), activation=activation) (c9)

    return outputs



def unet(input_shape = (128,128,6), output_channels=1, activation="sigmoid"):
    """

    :param input_shape: (tuple) dimensção do layer de entrada. default = (128,128,3).
    :param output_channels:(int) Número de canais de output
    :param activation: (string) função de ativação.
    :return: (tf.keras.models.Model) arquitetura de rede unet.
    """

    # definir a dimensão da camada de input
    inputs = tf.keras.layers.Input(shape=input_shape)

    # bloco encoder
    encoder_output, convs = encoder(inputs)

    # bottleneck
    bottle_neck = bottleneck(encoder_output)

    # decoder
    outputs = decoder(bottle_neck,convs, output_channels=output_channels,activation=activation)

    # montar o modelo
    model = tf.keras.Model(inputs, outputs)

    return model


def res_identity(x, f1,f2):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = keras.layers.Conv2D(f2,2, padding="same")(x) # this will be used for addition with the residual block
  f1, f2 = f1,f2

  #first block
  x = keras.layers.Conv2D(f1, kernel_size=(1, 1), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  #second block # bottleneck (but size kept same with padding)
  x = keras.layers.Conv2D(f1, kernel_size=(3, 3), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  # third block activation used after adding the input
  x = keras.layers.Conv2D(f2, kernel_size=(1, 1), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)

  # add the input
  x = keras.layers.Add()([x, x_skip])
  x = keras.layers.Activation("relu")(x)

  return x


def res_identity_downsample(x, f1,f2):
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = keras.layers.Conv2D(f2,2, padding="same",strides=(2,2))(x) # this will be used for addition with the residual block
  f1, f2 = f1,f2

  #first block
  x = keras.layers.Conv2D(f1, kernel_size=(1, 1),strides=(2,2), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  #second block # bottleneck (but size kept same with padding)
  x = keras.layers.Conv2D(f1, kernel_size=(3, 3), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  # third block activation used after adding the input
  x = keras.layers.Conv2D(f2, kernel_size=(1, 1), padding='same')(x)
  x = keras.layers.BatchNormalization()(x)

  # add the input
  x = keras.layers.Add()([x, x_skip])

  return x

def upsample_concat_block(filters,x, xskip):
    a = keras.layers.Conv2D(filters,(1,1),padding="same")(x)
    u = keras.layers.UpSampling2D((2, 2))(a)
    c = keras.layers.Concatenate()([u, xskip])
    return c


def resunet(image_size=(128,128,5)):
  inputs = keras.layers.Input(image_size)

  conv1 = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
  head = keras.layers.BatchNormalization()(conv1)
  head = keras.layers.Activation("relu")(head)

  conv2 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(head)
  head2 = keras.layers.BatchNormalization()(conv2)
  head2 = keras.layers.Activation("relu")(head2)
  head2 = keras.layers.MaxPooling2D((3,3),strides=(2,2),padding="same")(head2)

  res_unit_1 = res_identity(head2, 64,256)
  res_unit_1 = res_identity(res_unit_1,64,256)
  res_unit_1 = res_identity(res_unit_1,64,256)

  res_unit_2 = res_identity_downsample(res_unit_1,128,512)
  res_unit_2 = res_identity(res_unit_2,128,512)
  res_unit_2 = res_identity(res_unit_2,128,512)
  res_unit_2 = res_identity(res_unit_2,128,512)

  res_unit_3 = res_identity_downsample(res_unit_2,128,512)
  res_unit_3 = res_identity(res_unit_3,256,1024)
  res_unit_3 = res_identity(res_unit_3,256,1024)
  res_unit_3 = res_identity(res_unit_3,256,1024)
  res_unit_3 = res_identity(res_unit_3,256,1024)
  res_unit_3 = res_identity(res_unit_3,256,1024)
  res_unit_3 = res_identity(res_unit_3,256,1024)

  res_unit_4 = res_identity_downsample(res_unit_3,512,2048)
  res_unit_4 = res_identity(res_unit_4,512,2047)
  res_unit_4 = res_identity(res_unit_4,512,2048)

  u1 = upsample_concat_block(1024,res_unit_4,res_unit_3)

  u2 = upsample_concat_block(512,u1, res_unit_2)

  u3 = upsample_concat_block(256,u2,res_unit_1)

  u4 = upsample_concat_block(64,u3,conv2)

  u5 =  keras.layers.Conv2D(64,(1,1))(u4)
  u5 = keras.layers.UpSampling2D((2, 2))(u5)
  u5 = keras.layers.Add()([conv1,u5])

  outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)

  model = keras.models.Model(inputs, outputs)

  return model

def multiscale_fusion(input_data,filters):
  conv1 = keras.layers.Conv2D(filters,(3,3),dilation_rate=2,padding="same",activation="relu")(input_data)
  conv2 = keras.layers.Conv2D(filters,(3,3),dilation_rate=6,padding="same", activation="relu")(input_data)
  conv3 = keras.layers.Conv2D(filters,(3,3),dilation_rate=10,padding='same', activation="relu")(input_data)

  add1 = keras.layers.Add()([conv1,conv2])
  conv_1_1 = keras.layers.Conv2D(filters,(1,1), padding="same", activation="relu")(add1)
  add2 = keras.layers.Add()([conv_1_1,conv3])
  return add2

def expend_as(tensor,rep):
  my_repeat = keras.layers.Lambda(lambda x, repnum: keras.backend.repeat_elements(x, repnum, axis =3), arguments={"repnum":rep})(tensor)
  return my_repeat


def attention_module(input_data2,input_data1, inter_shape):
  gating = keras.layers.Conv2D(96, kernel_size=(1,1), padding="same")(input_data2)
  gating = keras.layers.BatchNormalization()(gating)
  gating = keras.layers.Activation("relu") (gating)

  shape_x = keras.backend.int_shape(input_data1)
  shape_g = keras.backend.int_shape(gating)

  theta_x = keras.layers.Conv2D(inter_shape,(2,2), strides=(2,2), padding="same")(input_data1)
  shape_theta_x = keras.backend.int_shape(theta_x)

  phi_g = keras.layers.Conv2D(inter_shape, (1,1), padding="same")(gating)
  upsample_g = keras.layers.Convolution2DTranspose(inter_shape, (3,3), strides=(shape_theta_x[1]//shape_g[1], shape_theta_x[2]// shape_g[2]),padding="same")(phi_g)

  concat_xg = keras.layers.Add()([upsample_g,theta_x])
  act_xg = keras.layers.Activation("relu")(concat_xg)
  psi = keras.layers.Conv2D(1,(1,1), padding="same")(act_xg)

  sigmoid_xg = keras.layers.Activation("sigmoid")(psi)
  shape_sigmoid = keras.backend.int_shape(sigmoid_xg)

  upsample_psi = keras.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2]//shape_sigmoid[2]))(sigmoid_xg)

  upsample_psi = expend_as(upsample_psi,shape_x[3])

  y = keras.layers.multiply([upsample_psi,input_data1])

  result = keras.layers.Conv2D(shape_x[3], (1,1), padding="same")(y)
  result_bn = keras.layers.BatchNormalization()(result)
  return result_bn

def res_net_block(input_data,filters,out_size):
  x = keras.layers.Conv2D(filters/2, kernel_size=(3,3), padding='same' )(input_data)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  x = keras.layers.Conv2D(filters, kernel_size=(3,3), activation='relu', padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation("relu")(x)

  x = keras.layers.Conv2D(filters, kernel_size=(1,1), activation='relu', padding='same')(x)
  x = keras.layers.BatchNormalization()(x)
  # x = keras.layers.Activation("relu")(x)

  x = keras.layers.Add()([x, input_data])

  x = keras.layers.Conv2D(out_size,(1,1), padding="same")(x)
  return x

def landsNet(image_size=(128,128,5)):
  inputs = keras.layers.Input(image_size)

  resBlock_1 = res_net_block(inputs,5,32)
  pooling_1 = keras.layers.MaxPooling2D(strides=2)(resBlock_1)

  resBlock_2 = res_net_block(pooling_1,32, 64)
  pooling_2 = keras.layers.MaxPooling2D(strides=2)(resBlock_2)

  resBlock_3 = res_net_block(pooling_2, 64,96)
  pooling_3 = keras.layers.MaxPooling2D(strides=2,name="pooling_3")(resBlock_3)

  multiscale_fusion1 = multiscale_fusion(pooling_3,96)

  conv1_1 = keras.layers.Conv2D(128,(1,1), padding="same",activation="relu")(multiscale_fusion1)

  #attention module -
  att_1 = attention_module(conv1_1,resBlock_3,96)

  # Upsample
  up1 = keras.layers.UpSampling2D()(conv1_1)


  # concatenate - attetion module + upsample
  concatenate_1 = keras.layers.Concatenate()([up1, att_1])


  resBlock_4 = res_net_block(concatenate_1,224,96)

  att_2 = attention_module(resBlock_4,resBlock_2,64)

  up2 = keras.layers.UpSampling2D()(resBlock_4)

  concatenate_2 = keras.layers.Concatenate()([up2, att_2])

  resBlock_5 = res_net_block(concatenate_2,160,64)

  att_3 = attention_module(resBlock_5,resBlock_1,32)

  up3 = keras.layers.UpSampling2D()(resBlock_5)

  concatenate_3 = keras.layers.Concatenate()([up3, att_3])

  resBlock_6 = res_net_block(concatenate_3,96,32)



  outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(resBlock_6)

  model = keras.models.Model(inputs, outputs)

  return model