import tensorflow as tf
from tensorflow.keras import backend as bkd
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
import pandas as pd
import seaborn as sns
import os, sys, glob, pathlib, configparser, random, math, json, time
from xtimer import Timer
import matplotlib.pyplot as plt
import utils, xmodels, constants
import gc

bkd.set_floatx('float32')
input_shape = constants.to_shape
model_struct= 'LFACon' # DADS_CNN, ALAS_DADS_CNN, ALASS_DADS_CNN, LFACon

##################################
version_no = 'v6.6'
##################################
fully_tained = True
# batch_size + eva_bz + n_val = 12
batch_size, n_val, val_ratio, eva_bz = 2, 2, 0.25, 1
n_epochs, n_mini_epochs = 500000, 10
n_batch = int(n_epochs/n_mini_epochs)
##################################
epo_patience = 1
baseline = 100
##################################
monitor_val = True
saving_type = 'weight' #model
reg_save_interv = 100
external_val = True
val_freq = 1 ###-

init_lr, end_lr, n_intv = 1e-6, 1e-6, 1 # 5-5 5-4 60e-6, 40e-6, 2
lr_change_intv, lr_repeat, lr_const_rmse = 200, True, 0 ###-
l_lrs = [round(lr,6) for lr in np.geomspace(init_lr, end_lr, num=n_intv, endpoint=True).tolist()] #linspace geomspace
##################################
start_lr_idx, lr_offset = 0, 0
##################################
no_wlbp = constants.no_wlbp
wts=[1,0.01,0.005,0.005] if not no_wlbp else [1,0.01,0.01] # [1,1,0.001,1]
loss_weights={'mos':wts[0], "spatial":wts[1], "angular_gdd":wts[2], "angular_wlbp":wts[3]} if not no_wlbp else {'mos':wts[0], "spatial":wts[1], "angular_gdd":wts[2]}

nal = True
ver_dir = utils.get_version_dir(model_struct)
_,model_root,s = utils.root_paths()
shared_prex = [f"{model_struct}-[{version_no}]-{str(n_epochs)}epo{str(n_mini_epochs)}-{str(int(n_epochs/n_mini_epochs))}bsz2=weight-p","--bst.json"]

#################################
shared_idx = ['6']
shared_paths = [shared_prex[0]+si+shared_prex[1] for si in shared_idx]
shared_paths=[ver_dir+sp for sp in shared_paths]


##################################
startup_model_path = ver_dir+"LFACon-[v6.6]-MPI-LFA.h5" 
##################################



model_prex = ver_dir+f"{model_struct}-[{version_no}]-{n_epochs}epo{n_mini_epochs}-{n_batch}bsz{batch_size}={saving_type}-p"
model_path, vn = utils.get_version_name(model_prex, ver_dir, '.json')





def build_train_model():
  print('Shared paths:',shared_paths)
  if model_struct == 'ALAS_DADS_CNN':
    historys = {"loss": [], "val_loss": [], "mos_loss": [], "val_mos_loss": [], "spatial_loss": [], "val_spatial_loss": [], "angular_gdd_loss": [], "val_angular_gdd_loss": [], "angular_wlbp_loss": [], "val_angular_wlbp_loss": []} if not no_wlbp else {"loss": [], "val_loss": [], "mos_loss": [], "val_mos_loss": [], "spatial_loss": [], "val_spatial_loss": [], "angular_gdd_loss": [], "val_angular_gdd_loss": []}
  else:
    historys = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}
    
  
  bst_list = {}
  bst_path = model_path+'--bst.h5'
  real_time_model_path= model_path+'--rt.h5'
  bst_paths = bst_path[:-3]+'.json'
  json.dump(bst_list, open(bst_paths, 'w'))
  print('lr list:',l_lrs)
  start_lr = l_lrs[start_lr_idx]
  if startup_model_path == None:
    model = xmodels.get_Xmodel(model_struct, input_shape, fully_tained,start_lr, loss_weights=loss_weights)
  else:
    assert startup_model_path.split(s)[-1].split('-')[0]==model_struct
    model = xmodels.get_Xmodel(model_struct, input_shape, fully_tained, start_lr, loss_weights=loss_weights)
    model.load_weights(startup_model_path)
    
  monitor = 'loss'
  if monitor_val:
    monitor='val_loss'
 
  es = tf.keras.callbacks.EarlyStopping(monitor=monitor,patience=epo_patience, verbose=2,restore_best_weights=True)
  csvlg =tf.keras.callbacks.CSVLogger(model_path+'.csv', separator=",", append=True)
  tnan = tf.keras.callbacks.TerminateOnNaN()
  cbks = [es, csvlg]
  # reduce_lr = ReduceLROnPlateau(monitor=monitor,factor=0.5,patience=5, min_lr=0.001)
  b_save_prex = model_path + '--bch_.h5'
  print('start training...')
  b_X = b_y = X_val = y_val = None
  list_with_replace_tr, list_with_replace_val = [], []

  old_lr = start_lr
  model.save_weights(real_time_model_path)

  for i in range(n_batch):
    b_n=i+1
    print('-'*50,'Bacth_' + str(b_n),'-'*50)
    b_save_path = b_save_prex[:-3] + str(b_n)+'.h5'
    print('lr:', model.optimizer.get_config())
    new_lr_idx = int((i+lr_offset)/lr_change_intv%n_intv) if lr_repeat else int((i+lr_offset)/lr_change_intv)
    new_lr = l_lrs[min(start_lr_idx + new_lr_idx, len(l_lrs)-1)]
    if new_lr!=old_lr:

      bkd.set_value(model.optimizer.learning_rate, new_lr)

      print('new lr:', new_lr)
      old_lr = new_lr
    tr_type = "train"

    if len(list_with_replace_tr) > 0:
      remain_eff=True
    b_X,b_y,_, _ = utils.generate_normalized_batch(model_struct=model_struct,batch_size=batch_size, tr_or_tt=tr_type, list_with_replace=list_with_replace_tr,normAL=nal, verbose=False) ###-
    if external_val:
      print('loading val...')
      X_val, y_val, list_with_replace_val, selected_val = utils.generate_normalized_batch(model_struct=model_struct,batch_size=n_val, tr_or_tt="train",list_with_replace=list_with_replace_val,normAL=nal, verbose=False)
      if model_struct == "CNN_4D":
        b_X = b_X.reshape((batch_size, 7, 7, 434, 434, 3), order='F')
        X_val = X_val.reshape((batch_size, 7, 7, 434, 434, 3), order='F')
        # print(b_X.shape)

    model.save_weights(real_time_model_path)

    if not external_val:
      h = model.fit(b_X, b_y, epochs=n_mini_epochs, validation_split=val_ratio,  validation_freq=val_freq, verbose=1, batch_size=batch_size,  use_multiprocessing=constants.multip, workers=constants.n_workers, callbacks=cbks)
    else:
      h = model.fit(b_X, b_y, epochs=n_mini_epochs, validation_data=(X_val,y_val), validation_freq=val_freq, verbose=1, batch_size=batch_size, use_multiprocessing=constants.multip, workers=constants.n_workers, callbacks=cbks)
    
    if (reg_save_interv > 0 and b_n % reg_save_interv == 0) or b_n == 1:
      if saving_type == 'weight':
        model.save_weights(b_save_path)
      else:
        model.save(b_save_path)
    

    print(f'List size - tr: {len(list_with_replace_tr)}, val: {len(list_with_replace_val)}')
    historys = merge_save_history(historys, h.history)
    print(b_n, "out of", n_batch, "batches completed")
    t.lap()

    model.save_weights(real_time_model_path)
    t.lap()
  return model, historys




def merge_save_history(hs, h):
  l=len(h['loss'])
  for k,v in h.items():
    if len(v)!= l:
      hs[k]+=([0]*l)
    else:
      hs[k]+=v
  json.dump(hs, open(model_path+'.json', 'w'))
  return hs



def save_pic_results(historys):
  mse_label = 'mse' if model_struct == "ALAS_DADS_CNN" else 'loss'
  plt.plot(np.sqrt(historys[mse_label]), label='rmse')
  plt.plot(np.sqrt(historys['val_'+mse_label]), label = 'val_rmse')
  plt.xlabel('Epoch')
  plt.ylabel('RMSE')
  plt.ylim([0, 10])
  plt.legend(loc='lower right')

  last_n=100
  print('='*10,'Training Results','='*10)
  tr_rmse = np.mean(np.sqrt(historys[mse_label])[-last_n:])
  print(f"Mean RMSE of last {last_n} epoches: {tr_rmse}")
  print('='*10,'Validation Results','='*10)
  val_rmse=np.mean(np.sqrt(historys['val_'+mse_label])[-last_n:])
  print(f"Mean RMSE of last {last_n} epoches: {val_rmse}")
  print('='*10,'Test Results','='*10)
  loss, mae, mse, rmse, srcc, lcc = utils.evaluate_tt((model_path,model),batch_size=eva_bz,normAL=nal)
  print("-"*10,'loss: {:5.4f}, mae: {:5.4f}, mse: {:5.4f}'.format(loss, mae, mse),"-"*10)
  print("+"*10,"TEST RMSE: {:5.4f}".format(rmse),"+"*10, '\n')
  total_time = int(t.total_t())
  plt.title('RMSE tr:{:.4f}val:{:.4f}tt:{:.4f} {}s SRCC:{:.4f}LCC:{:.4f}'.format(tr_rmse,val_rmse,rmse,total_time, srcc, lcc))
  plt.savefig(model_path+'.png', dpi=1200)  #os:{} ,sys.platform




if __name__ == "__main__":
  # main
  print('='*50)
  t = Timer()
  t.start()
  if len(sys.argv) > 1:
    if sys.argv[1][0] == 'd':
      if sys.argv[1] == 'dcsc':
        strategy = tf.distribute.experimental.CentralStorageStrategy()
        # exit(), compute_devices=['/job:localhost/replica:0/task:0/device:GPU:1'], parameter_device=None
      elif sys.argv[1] == 'dm':
        strategy = tf.distribute.MirroredStrategy()
        # cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
      elif sys.argv[1] == 'dcsg':
        strategy = tf.distribute.experimental.CentralStorageStrategy(parameter_device='/job:localhost/replica:0/task:0/device:GPU:0')
      elif sys.argv[1] == 'dmh':
        strategy = tf.distribute.MirroredStrategy(devices=['/job:localhost/replica:0/task:0/device:GPU:0','/job:localhost/replica:0/task:0/device:GPU:1'],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
      else:
        strategy = tf.distribute.experimental.CentralStorageStrategy()
      # train with the strategy
      with strategy.scope():
        model, historys = build_train_model()
    elif sys.argv[1][0] == 'g':
      # tf.config.set_soft_device_placement(True)
      gpu_prex = '/job:localhost/replica:0/task:0/device:GPU:'
      with tf.device('/device:CPU:0'):
        with tf.device(gpu_prex+sys.argv[1][1]):
          model, historys = build_train_model()
    elif sys.argv[1][0] == 'c':
      with tf.device('/device:CPU:0'):
        model, historys = build_train_model()
  else:
    model, historys = build_train_model()

  if saving_type == 'weight':
    model.save_weights(model_path+'.h5')
  else:
    model.save(model_path+'.h5')
  # history_dict_list = json.load(open(model_path+'.json', 'r'))
  save_pic_results(historys)
  print('='*100)
  t.stop()
