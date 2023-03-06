import tensorflow as tf
from tensorflow.keras import layers, models, Model, activations, initializers
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import constants
from tensorflow.keras.utils import plot_model
# from conv4d import conv4d
# from IPython.display import Image
# from IPython.display import clear_output



def ini_kern():
  return initializers.GlorotNormal()


def xrelu(X):
  return relu_layer(X, relu_type='swish')


def l_norm(X):
  return layers.LayerNormalization()(X)




def bnorm_layer(X, no_bn=True):
  if no_bn:
    return X
  return layers.BatchNormalization(axis=-1,epsilon=1e-3,momentum=0.6)(X)



#(-2,6) (-5,10) (-10,20)
def relu_layer(X,mxv=20,nslp=0,thd=-10,alpha=0.2,name=None,relu_type='linear'):
  if relu_type=='relux':
    return layers.ReLUx(max_value=mxv, negative_slope=nslp, threshold=thd)(X)
  elif relu_type=='leaky':
    return layers.LeakyReLU(alpha=alpha)(X)
  elif relu_type=='linear':
    return X
  elif relu_type=='selu':
    return activations.selu(X)
  elif relu_type=='swish':
    return activations.swish(X)
  elif relu_type=='elu':
    return activations.elu(X)
  elif relu_type=='gelu':
    return activations.gelu(X)
  elif relu_type=="tanh":
    return activations.tanh(X)
  elif relu_type=="softsign":
    return activations.softsign(X)
  else:
    return activations.relu(X)
  



def scaled_dot_product_attention(q, k, v, mask=None):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights



class LFA_self(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, embed="conv", d1_pre, d1_post, sqrt_n, n_filters, inshape):
    super(LFA_self, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.embed = embed

    self.d1_pre, self.d1_post, self.sqrt_n, self.n_filters = d1_pre, d1_post, sqrt_n, n_filters

    self.inshape = inshape

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    if embed == "dsc":
      self.wq = LF_DSC(n_filters=n_filters, n_sais=self.d1_post, input_shape=inshape)
      self.wk = LF_DSC(n_filters=n_filters, n_sais=self.d1_pre, input_shape=inshape)
      self.wv = LF_DSC(n_filters=n_filters, n_sais=self.d1_pre, input_shape=inshape)

      self.wout = LF_DSC(n_filters=n_filters, n_sais=self.d1_post, input_shape=inshape)
    elif embed == "conv":
      self.wq = layers.Conv3D(n_filters, 1,padding='same')
      self.wk = layers.Conv3D(n_filters, 1,padding='same')
      self.wv = layers.Conv3D(n_filters, 1,padding='same')

      self.wout = layers.Conv3D(n_filters, 1,padding='same')
    elif embed == "conv_dsc":
      self.wq = layers.Conv3D(n_filters, 1,padding='same')
      self.wk = layers.Conv3D(n_filters, 1,padding='same')
      self.wv = layers.Conv3D(n_filters, 1,padding='same')

      self.wout = LF_DSC(n_filters=n_filters, n_sais=self.d1_post, input_shape=inshape)
    else:
      self.wq = tf.keras.layers.Dense(d_model)
      self.wk = tf.keras.layers.Dense(d_model)
      self.wv = tf.keras.layers.Dense(d_model)
      self.wout = tf.keras.layers.Dense(d_model)
      


    
# credits https://pytorch.org/tutorials/beginner/transformer_tutorial.html
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]

    if self.embed == "nn":
      q = self.wq(q)  # (batch_size, seq_len, d_model)
      k = self.wk(k)  # (batch_size, seq_len, d_model)
      v = self.wv(v)  # (batch_size, seq_len, d_model)
    else:
      q = self.wq(q)  # (batch_size, seq_len, d_model)
      k = self.wk(k)  # (batch_size, seq_len, d_model)
      v = self.wv(v)  # (batch_size, seq_len, d_model)

      q = reshape2d(q)
      k = reshape2d(k)
      v = reshape2d(v)
      

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    if self.embed != "nn":
      concat_attention = reshape4d(concat_attention, self.d1_post, self.sqrt_n, self.n_filters)

    output = self.wout(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights



def list_multi(l):
  m = 1
  for i in l:
    m *= i
  return m



def reshape2d(X):
  # print(X.get_shape().as_list())
  # print(tf.shape(X)[0])
  tshape = X.get_shape()
  last_dim = list_multi(tshape[2:])
  # print(last_dim)
  return tf.reshape(X, (tf.shape(X)[0], tshape[1], last_dim))



def reshape4d(X, d1, sqrt_n, n_filters):
  # print(X.get_shape().as_list())
  # print(tf.shape(X)[0])
  tshape = X.get_shape()
  # last_dim = sum(tshape[2:])
  # X=tf.expand_dims(X, -1)
  # print(tshape)
  return tf.reshape(X, (tf.shape(X)[0], d1, sqrt_n, sqrt_n, n_filters))



class LF_DSC(tf.keras.layers.Layer):
  def __init__(self,*, n_filters=16,filter_size=(3,3), n_sais=49, strides=1, residual=True, include_act=True, input_shape):
    super(LF_DSC, self).__init__()

    self.n_filters= n_filters
    self.filter_size=filter_size
    self.n_sais= n_sais
    self.strides= strides
    self.residual= residual
    self.include_act= include_act

    self.layer_1=layers.Conv3D(n_filters, 1,padding='same')

    self.layers_2 = [layers.DepthwiseConv2D(filter_size, strides=strides, padding='same',input_shape=input_shape[2:]) for i in range(n_sais)]

    self.layer_3 = layers.Conv3D(n_filters, 1,padding='same',activation='linear')


  def call(self, X):

    inputs = X
    X=self.layer_1(X)
    # X=xrelu(X)

    if self.residual:
      self.strides=1
    stack = []
    for i in range(self.n_sais):
      X1= X[:,i,:,:,:]
      X1 = self.layers_2[i](X1)
      stack.append(X1)
    X=tf.stack(stack, axis=1)
    X=xrelu(X)

    X=self.layer_3(X)
    # X=xrelu(X)

    if self.residual:
      X = layers.Add()([X, inputs])
    return X



def LFDepthwiseConvBlock(inputs, n_filters=16,filter_size=(3,3), n_sais=49, strides=2, residual=False, include_act=True):
  X=layers.Conv3D(n_filters, 1,padding='same',activation='linear')(inputs)
  # if include_act: X=xrelu(X)

  if residual:
    strides=1
  stack = []
  for i in range(n_sais):
    X1= X[:,i,:,:,:]
    X1 = layers.DepthwiseConv2D(filter_size, strides=strides, padding='same',input_shape=X.shape[2:], kernel_initializer=ini_kern(),activation='linear')(X1)
    stack.append(X1)
  X=tf.stack(stack, axis=1)
  if include_act: X=xrelu(X)

  X=layers.Conv3D(n_filters, 1, padding='same',activation='linear')(X)
  # if include_act: X=xrelu(X)

  if residual:
    X = layers.Add()([X, inputs])
  
  X = l_norm(X)
  return X



def LFAngularConvBlock(inputs, n_filters=8,filter_size=(7,7,4,4), n_sais=49, dilations=(7,1,1), residual=False):# 3, 3
  filter_size_1 = filter_size[:1]+filter_size[2:]
  X=layers.Conv3D(n_filters, kernel_size=filter_size_1, padding='same', kernel_initializer=ini_kern())(inputs)
  X=xrelu(X)

  filter_size_2=filter_size[1:]
  X=layers.Conv3D(n_filters, kernel_size=filter_size_2, dilation_rate=dilations, padding='same', kernel_initializer=ini_kern())(X)
  X=xrelu(X)
  return X


def LFACon_self(X, sqrt_n, resid=True, resid_con=True, n_filters=3, num_heads=8, verbose=False, strides=4, filter_size=(3,3), d1=49, with_con=True):
  
  if verbose: print('-'*20)
  if verbose: print(X.get_shape())
  if with_con: X=LFDepthwiseConvBlock(X, n_filters=n_filters, residual=resid_con, strides=strides, filter_size=filter_size, n_sais=d1)


  if resid:
    X0 = X

  # if verbose: print(X.get_shape())
  # X = reshape2d(X)
  
  if verbose: print(X.get_shape())
  lfa_self = LFA_self(d_model=int(sqrt_n**2*n_filters), num_heads=num_heads, d1_pre=d1, d1_post=d1, sqrt_n=sqrt_n, n_filters=n_filters, inshape=X.shape)
  X, attn = lfa_self(X, k=X, q=X)
  if verbose: print(X.get_shape())
  
  # X=xrelu(X)
  # X = reshape4d(X, d1=d1, sqrt_n=sqrt_n, n_filters=n_filters)

  X=layers.Conv3D(n_filters, 1, padding='same',activation='linear')(X)
  # X=xrelu(X)
  if resid: X = layers.Add()([X0, X])
  if verbose: print(X.get_shape())

  X = l_norm(X)
  return X



def LFACon_grid(X, sqrt_n, resid=True, resid_con=True, n_filters=3, num_heads=8, verbose=False, strides=1, filter_size=(3,3), n_sais=49, with_con=True):
  if verbose: print('-'*20)
  if verbose: print(X.get_shape())
  if with_con: X=LFDepthwiseConvBlock(X, n_filters=n_filters, residual=resid_con, strides=strides, filter_size=filter_size)
  if verbose: print(X.get_shape())

  # X = reshape2d(X)
  # if verbose: print(X.get_shape())

  target_i = [8, 10, 12, 22, 24, 26, 36, 38, 40]
  stack = []
  lfa_self = lfa_self = LFA_self(d_model=int(sqrt_n**2*n_filters), num_heads=num_heads, d1_pre=49, d1_post=9, sqrt_n=sqrt_n, n_filters=n_filters, inshape=X.shape)
  for i in target_i:
    X1= X[:,i,:]
    
    stack.append(X1)
  X2=tf.stack(stack, axis=1)

  X, _ = lfa_self(X, k=X, q=X2)
  # X=xrelu(X)

  if verbose: print(X.get_shape())
  # X = reshape4d(X, 9, sqrt_n=sqrt_n, n_filters=n_filters)

  X=layers.Conv3D(n_filters, 1, padding='same',activation='linear')(X)
  # X=xrelu(X)

  if verbose: print(X.get_shape())

  if resid: X = layers.Add()([X2, X])
  
  X = l_norm(X)
  return X



def LFACon_centre(X, sqrt_n, resid=True, resid_con=True, n_filters=3, num_heads=8, verbose=False, strides=1, filter_size=(3,3), n_sais=9, with_con=True):
  if verbose: print('-'*20)
  if verbose: print(X.get_shape())
  if with_con: X=LFDepthwiseConvBlock(X, n_filters=n_filters, residual=resid_con, strides=strides, filter_size=filter_size, n_sais=9)

  if verbose: print(X.get_shape())
  # X = reshape2d(X)
  # if verbose: print(X.get_shape())

  lfa_self = lfa_self = LFA_self(d_model=int(sqrt_n**2*n_filters), num_heads=num_heads, d1_pre=9, d1_post=1, sqrt_n=sqrt_n, n_filters=n_filters, inshape=X.shape)
  Xc= X[:,4,:]
  Xc = tf.expand_dims(Xc, axis=1)
  # if verbose: print("Xc",Xc.get_shape())

  X, _ = lfa_self(X, k=X, q=Xc)
  if verbose: print(X.get_shape())

  # X=xrelu(X)
  # X = reshape4d(X, 1, sqrt_n=sqrt_n, n_filters=n_filters)
  X=layers.Conv3D(n_filters, 1, padding='same',activation='linear')(X)
  # X=xrelu(X)
  if verbose: print(X.get_shape())

  if resid: X = layers.Add()([Xc, X])

  X = l_norm(X)
  return X


def spatial_reduction_1(X):
  X=layers.Conv3D(3, kernel_size=(1,3,3), strides=(1,2,2), padding='same',activation='linear')(X)
  X=xrelu(X)

  # spatial reduction
  X=LFDepthwiseConvBlock(X, n_filters=3, residual=True, strides=1, filter_size=(3,3))
  X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=LFDepthwiseConvBlock(X, n_filters=6, residual=False, strides=1, filter_size=(3,3))

  X=LFDepthwiseConvBlock(X, n_filters=6, residual=True, strides=1, filter_size=(3,3))
  X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=LFDepthwiseConvBlock(X, n_filters=12, residual=False, strides=1, filter_size=(3,3))

  X=LFDepthwiseConvBlock(X, n_filters=12, residual=True, strides=1, filter_size=(3,3))
  X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=LFDepthwiseConvBlock(X, n_filters=24, residual=False, strides=1, filter_size=(3,3))

  # print(X.get_shape())
  # X = l_norm(X)
  return X


def channel_expansion(X):
  # spatial reduction
  X=LFDepthwiseConvBlock(X, n_filters=96, residual=True, strides=1, filter_size=(3,3), n_sais=1)
  X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=LFDepthwiseConvBlock(X, n_filters=192, residual=False, strides=1, filter_size=(3,3), n_sais=1)

  X=LFDepthwiseConvBlock(X, n_filters=192, residual=True, strides=1, filter_size=(3,3), n_sais=1)
  X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=LFDepthwiseConvBlock(X, n_filters=384, residual=False, strides=1, filter_size=(3,3), n_sais=1)

  return X

  

#Total params: 
def get_LFACon(input_shape, verbose=False):
  inputs = layers.Input(shape = input_shape,name='lfi_input')

  X = inputs
  
  X = spatial_reduction_1(X)

  X = LFACon_self(X, 28, resid=True, resid_con=True, n_filters=24, num_heads=8, verbose=verbose, strides=1, filter_size=(3,3), with_con=True)
  X = LFACon_self(X, 28, resid=True, resid_con=True, n_filters=24, num_heads=8, verbose=verbose, strides=1, filter_size=(3,3), with_con=True)
  X = LFACon_self(X, 28, resid=True, resid_con=True, n_filters=24, num_heads=8, verbose=verbose, strides=1, filter_size=(3,3), with_con=True)

  X=LFACon_grid(X, 28, verbose=verbose, n_filters=48, num_heads=8, resid=True, resid_con=False)
  X = LFACon_self(X, 28, resid=True, resid_con=True, n_filters=48, num_heads=8, verbose=verbose, strides=1, filter_size=(3,3), with_con=True, d1=9)
  X = LFACon_self(X, 28, resid=True, resid_con=True, n_filters=48, num_heads=8, verbose=verbose, strides=1, filter_size=(3,3), with_con=True, d1=9)

  X=LFACon_centre(X, 28, verbose=verbose, n_filters=96, num_heads=8, resid=True, resid_con=False)

  X=channel_expansion(X)

  # X=layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(X)
  X=layers.Conv3D(768, kernel_size=(1,1,1), strides=(1,1,1), padding='same', activation='linear')(X)
  # X=xrelu(X)


  X= layers.Flatten()(X)
  X= layers.Dense(256)(X)

  pred_mos = layers.Dense(1, name='mos')(X)

  predictions = [pred_mos]
  model = models.Model(inputs=inputs,outputs=predictions)
  if verbose:
    model.summary()
  return model



def get_Xmodel(model_struct, input_shape, fully_tained=True, learning_rate=0.001, loss_weights=[0.1,0.1,1]):
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,amsgrad=True)#,clipnorm=6.0, clipvalue=0.5
  if model_struct == 'DADS_CNN':
    model=get_DADS_CNN(input_shape,False)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
  elif model_struct == 'ALAS_DADS_CNN':
    model=get_ALAS_DADS_CNN(input_shape,False)
    loss={'mos':'mse', "spatial":'mse', "angular_gdd":'mse', "angular_wlbp":'mse'} if not constants.no_wlbp else {'mos':'mse', "spatial":'mse', "angular_gdd":'mse'}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
  elif model_struct == 'ALASS_DADS_CNN':
    pass
  elif model_struct == 'LFACon':
    model=get_LFACon(input_shape,False)
    # print("+++++++++++++++++")
    # print(model)
    loss={'mos':'mse'}
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
  else:
    if model_struct == "CNN_4D":
      model=get_CNN_4D(input_shape,False)
    else:
      model=get_SM(model_struct, input_shape,False)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
  return model



def remodel(model_struct, input_shape, model, fully_tained=True, learning_rate=0.001, loss_weights=[0.1,0.1,1]):
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,amsgrad=True)#,clipnorm=6.0, clipvalue=0.5
  if model_struct == 'DADS_CNN':
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
  elif model_struct == 'ALAS_DADS_CNN':
    loss={'mos':'mse', "spatial":'mse', "angular_gdd":'mse', "angular_wlbp":'mse'} if not constants.no_wlbp else {'mos':'mse', "spatial":'mse', "angular_gdd":'mse'}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
  elif model_struct == 'ALASS_DADS_CNN':
    pass
  else:
    loss={'mos':'mse', "spatial":'mse', "angular_gdd":'mse', "angular_wlbp":'mse'} if not constants.no_wlbp else {'mos':'mse', "spatial":'mse', "angular_gdd":'mse'}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights)
  return model



def draw_model(model, version="2.3", model_struct="ALAS_DADS_CNN"):
  plot_model(model,rankdir='LR',show_shapes=False,show_layer_names=False,to_file=f"{model_struct}_{version}.1.png",dpi=300)
  plot_model(model,rankdir='LR',show_shapes=False,show_layer_names=True,to_file=f"{model_struct}_{version}.2.png",dpi=300)
  plot_model(model,rankdir='LR',show_shapes=True,show_layer_names=False,to_file=f"{model_struct}_{version}.3.png",dpi=300)
  plot_model(model,rankdir='LR',show_shapes=True,show_layer_names=True,to_file=f"{model_struct}_{version}.4.png",dpi=300)



if __name__ == "__main__":
  # model = get_DADS_CNN(constants.to_shape,True)
  # model = get_ALAS_DADS_CNN(constants.to_shape,True)
  # draw_model(model)
  # model = get_CNN_4D(constants.to_shape,True) # 374,072
  # model = get_SM("ASC",constants.to_shape,True) # 20,222
  # model = get_SM("DSC",constants.to_shape,True) # 25,232
  # model = get_SM("DSC_ASC",constants.to_shape,True) # 45,452
  model = get_LFACon(constants.to_shape, verbose=True)
  
  
 
