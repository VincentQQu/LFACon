import numpy as np
# from sklearn.model_selection import train_test_split
import os, glob, sys, constants, random, xmodels, json, math
from xtimer import Timer
import tensorflow as tf
from PIL import Image
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from csv import writer
dn = constants.dataset_name
af = constants.aug_factor
r1 = constants.reduce_1
no_wlbp = constants.no_wlbp
img_format = constants.img_format
# lcc_test, _ = pearsonr([1,1,1],[1,1,1])



def generate_normalized_batch(model_struct='DADS_CNN', batch_size=5, tr_or_tt="train", list_with_replace=None, norm_method="stand", normAL=True, verbose=False):
  if verbose:
    t = Timer()
    t.start()
    print("generating batch...")
  dataset_root, _, s = root_paths()
  if af =='x32':
    read_dir = dataset_root+ dn+"-tr-tt" + s
  else:
    read_dir = dataset_root+ dn+"-tr-tt-"+af + s
  if list_with_replace == None:
    read_img_paths = glob.glob(read_dir+tr_or_tt+"=*."+img_format, recursive=False)
    smp = random.sample(read_img_paths, batch_size)
  else:
    if len(list_with_replace)<batch_size:
      list_with_replace=glob.glob(read_dir+tr_or_tt+"=*."+img_format, recursive=False)
    random.shuffle(list_with_replace)
    smp=list_with_replace[:batch_size]
    list_with_replace=list_with_replace[batch_size:]
  if normAL:
    sam_labels_path = dataset_root+ dn+"-labels" + s +dn+'_SAM_labels.json'
  else:
    sam_labels_path = dataset_root+ dn+"-labels" + s +dn+'_SAM_labels.json'
  labels = json.load(open(sam_labels_path, 'r'))
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  b_X_shape = (batch_size,to_shape[0],to_shape[1],to_shape[2],to_shape[3])
  b_X = np.zeros(b_X_shape, dtype=np.float32)
  if model_struct == 'ALAS_DADS_CNN':
    b_y={}
    b_y_mos = np.zeros((batch_size,1), dtype=np.float32)
    b_y_spa = np.zeros((batch_size,constants.n_spa_feats), dtype=np.float32)
    b_y_ang_gdd = np.zeros((batch_size,constants.n_ang_feats_gdd), dtype=np.float32)
    if not no_wlbp:
      b_y_ang_wlbp = np.zeros((batch_size,constants.n_ang_feats_wlbp), dtype=np.float32)
  else:
    b_y = np.zeros((batch_size,1), dtype=np.float32)
  for i, p in enumerate(smp):
    name=p.split(s)[-1][:-4]
    # if dn=="Win5-LID":
    #   if "train" in p:
    #     name = p.split(s)[-1][6:-4]
    #   else:
    #     name = p.split(s)[-1][5:-4]
      # if af == "x8" :
      #   name += "+crop0"
    image = Image.open(p)
    data = np.asarray(image)
    image.close()
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape).astype('float32')
    b_X[i] = normalize(a, norm_method=norm_method)
    mos = labels[name]['mos']
    if verbose or True:
      print(tr_or_tt,name,mos)
    if model_struct == 'ALAS_DADS_CNN':
      b_y_mos[i] = mos
      b_y_spa[i] = np.array(labels[name]['spatial'])
      b_y_ang_gdd[i] = np.array(labels[name]['angular_gdd'])
      if not no_wlbp:
        b_y_ang_wlbp[i] = np.array(labels[name]['angular_wlbp'])
    else:
      b_y[i] = mos
  if model_struct == 'ALAS_DADS_CNN':
    # if r1:
      # b_X={'lfi_input': b_X[:,to_shape[0]-1,to_shape[1]-1,:,:,:]}
    b_X={'lfi_input': b_X}
    b_y['mos'] = b_y_mos
    b_y['spatial'] = b_y_spa
    b_y['angular_gdd'] = b_y_ang_gdd
    if not no_wlbp:
      b_y['angular_wlbp'] = b_y_ang_wlbp
  check_nan=(hasnan(b_X,model_struct),hasnan(b_y,model_struct))
  if verbose:
    print(check_nan)
  if verbose:
    print("success! - generate_normalized_batch")
    t.stop()
  return b_X, b_y, list_with_replace, smp




def evaluate_tt(model_tuple, batch_size=10, normAL=True,verbose=False):
  t = Timer()
  t.start()
  model_path,model = model_tuple
  _, _, s = root_paths()
  model_name = model_path.split(s)[-1]
  print('='*20,f'Model: {model_name}','='*20)
  results_path = get_version_dir() + 'results.csv'
  if os.path.exists(results_path) and batch_size !=1:
    results = pd.read_csv(results_path)
    selected_results = results.index[results['Model']==model_name].tolist()
    if len(selected_results) > 0:
      print("The evaluation result exists!")
      result_i = selected_results[0]
      loss_all, mae_all, mse_all = results['Loss'][result_i], results['MAE'][result_i], results['MSE'][result_i]
      rmse, srcc, lcc = results['RMSE'][result_i], results['SRCC'][result_i], results['LCC'][result_i]
      show_evaluation_results(loss_all, mae_all, mse_all, rmse, srcc, lcc)
      t.stop()
      return loss_all, mae_all, mse_all, rmse, srcc, lcc

  model_struct = model_name.split('-')[0]
  if model == None:
    if 'weight' in model_path:
      model = xmodels.get_Xmodel(model_struct,constants.to_shape,loss_weights=1)
      model.load_weights(model_path)
    else:
      model = tf.keras.models.load_model(model_path)
  if verbose:
    model.summary()
    print("generating test set...")
    t.lap()
  b_paths=generate_test_batches(batch_size=batch_size,model_struct=model_struct, normAL=normAL)
  if verbose:
    print("evaluating...")
    t.lap()
  loss_all, mae_all, mse_all = 0,0,0
  y_preds_mos,y_trues_mos,y_names = [],[],[]
  model_outputs={}
  # print(b_paths)
  for b_p in b_paths:
    # if model_struct=='ALAS_DADS_CNN': 
    #   b=np.load(b_p,allow_pickle=True)
    #   b_X=b[()]["b_X"]
    #   b_y=b[()]["b_y"]
    #   # print('b_X',b_X)
    #   # print(b_X["lfi_input"].shape)
    #   l=b_X["lfi_input"].shape[0]
    #   # print(b_X["lfi_input"][0].shape)
    #   y_pred = model.predict(b_X)
    #   y_name = b_p.split(s)[-1][:-4]
    #   y_names.append(y_name)
    #   model_outputs[y_name]={}
    #   y_pred_sqzd=sqz_y_pred(y_pred)
    #   # print('y_pred_sqzd',y_pred_sqzd)
    #   res_dict = model.evaluate(b_X, b_y, verbose=verbose,batch_size=l, use_multiprocessing=constants.multip, workers=constants.n_workers,return_dict=True)
    #   loss, mae, mse=res_dict['mos_loss'],-1,res_dict['mos_loss']
    #   b_y_sqzd=sqz_b_y(b_y)
    #   # print('b_y_sqzd',b_y_sqzd)
    #   if batch_size==1 or b_p[-7:-4]=="bz1":
    #     y_preds_mos.append(y_pred_sqzd['mos'])
    #     y_trues_mos.append(b_y_sqzd['mos'])
    #     print('Predicted:',y_pred_sqzd['mos'],'True:', b_y_sqzd['mos'])
    #   else:
    #     y_preds_mos+=y_pred_sqzd['mos']
    #     y_trues_mos+=b_y_sqzd['mos']
    #   model_outputs[y_name]['y_pred']=y_pred_sqzd
    #   model_outputs[y_name]['y_true']=b_y_sqzd
    # else:
    with np.load(b_p) as b:
      print(b_p)
      b_X=b["b_X"]
      b_y=b["b_y"]
    if model_struct == 'CNN_4D':
      b_X = b_X.reshape((batch_size, 7, 7, 434, 434, 3), order='F')
    l=b_X.shape[0]
    y_pred = model.predict(b_X)
    y_name = b_p.split(s)[-1][:-4]
    y_names.append(y_name)
    res_dict = model.evaluate(b_X, b_y, verbose=verbose,batch_size=l, use_multiprocessing=constants.multip, workers=constants.n_workers,return_dict=True)
    if "mae" in res_dict:
      loss, mae, mse=res_dict['loss'],res_dict['mae'],res_dict['loss']
    else:
      loss, mae, mse=res_dict['loss'],res_dict['loss'],res_dict['loss']
    y_pred_sqzd=np.squeeze(y_pred).tolist()
    b_y_sqzd=np.squeeze(b_y).tolist()
    model_outputs[y_name]={}
    model_outputs[y_name]['y_pred']=y_pred_sqzd
    model_outputs[y_name]['y_true']=b_y_sqzd
    if batch_size==1 or "bz1." in b_p:
      y_preds_mos.append(y_pred_sqzd)
      y_trues_mos.append(b_y_sqzd)
      print(y_pred_sqzd, b_y_sqzd)
    else:
      y_preds_mos+=y_pred_sqzd
      y_trues_mos+=b_y_sqzd
    loss_all+=(loss*l)
    mae_all+=(mae*l)
    mse_all+=(mse*l)
    if verbose:
      t.lap()

  n_tt = constants.n_tt
  loss_all = loss_all/n_tt
  mae_all = mae_all/n_tt
  mse_all = mse_all/n_tt
  rmse = np.sqrt(mse_all)
  srcc, _ = spearmanr(y_preds_mos,y_trues_mos)
  _, _ = pearsonr(y_preds_mos,y_trues_mos)
  lcc, _ = pearsonr(y_preds_mos,y_trues_mos)
  show_evaluation_results(loss_all, mae_all, mse_all, rmse, srcc, lcc)
  new_row = [model_name, loss_all,mae_all,mse_all,rmse,srcc,lcc]
  df = pd.DataFrame([new_row], columns=["Model","Loss", "MAE", "MSE", "RMSE", "SRCC", "LCC"])
  if not os.path.exists(results_path):
    df.to_csv(results_path, index=False)
  else:
    append_list_as_row(results_path, new_row)
  save_model_outputs(model_outputs,y_preds_mos, y_trues_mos, model_name,s, y_names)
  print("evaluation result was saved.")
  if verbose:
    print("success! - evaluation")
  t.stop()
  return loss_all, mae_all, mse_all, rmse, srcc, lcc




def generate_test_batches(batch_size=10,norm_method="stand",model_struct='DADS_CNN',save_dir=None, normAL=True, verbose=False):
  dataset_root, _, s = root_paths()
  if verbose:
    t=Timer()
    t.start()
  if af =='x32':
    read_dir = dataset_root+ dn+"-tr-tt" + s
    save_dir=dataset_root+dn+"-tensor"+s
  else:
    read_dir = dataset_root+ dn+"-tr-tt-"+af + s
    save_dir=dataset_root+dn+"-tensor-"+af+s
  # model_struct = 'DADS_CNN' if model_struct != 'ALAS_DADS_CNN' else "ALAS_DADS_CNN"
  save_dir = save_dir+f"{model_struct}-tt-bz{batch_size}-{norm_method}-nal{int(normAL)}-nw{int(no_wlbp)}"+s
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  else:
    print(save_dir, "exists!")
    if model_struct == 'ALAS_DADS_CNN':
      save_paths = glob.glob(save_dir+"*.npy", recursive=False)
    else:
      save_paths = glob.glob(save_dir+"*.npz", recursive=False)
    return save_paths
  read_img_paths = glob.glob(read_dir+"test=*."+img_format, recursive=False)
  if normAL:
    sam_labels_path = dataset_root+ dn+"-labels" + s +dn+'_SAM_labels.json'
  else:
    sam_labels_path = dataset_root+ dn+"-labels" + s +dn+'_SAM_labels.json'
  labels = json.load(open(sam_labels_path, 'r'))
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  chunks = [read_img_paths[i:i+batch_size] for i in range(0, len(read_img_paths), batch_size)]
  for i, c in enumerate(chunks):
    l=len(c)
    b_X_shape = (l,to_shape[0],to_shape[1],to_shape[2],to_shape[3])
    b_X = np.zeros(b_X_shape, dtype=np.float32)
    # if model_struct == 'ALAS_DADS_CNN':
    #   b_y={}
    #   b_y_mos = np.zeros((l,1), dtype=np.float32)
    #   b_y_spa = np.zeros((l,constants.n_spa_feats), dtype=np.float32)
    #   b_y_ang_gdd = np.zeros((l,constants.n_ang_feats_gdd), dtype=np.float32)
    #   if not no_wlbp:
    #     b_y_ang_wlbp = np.zeros((l,constants.n_ang_feats_wlbp), dtype=np.float32)
    # else:
    b_y = np.zeros((l,1), dtype=np.float32)
    for j, p in enumerate(c):
      name = p.split(s)[-1][:-4]
      # if dn=="Win5-LID":
      #   if "train" in p:
      #     name = p.split(s)[-1][6:-4]
      #   else:
      #     name = p.split(s)[-1][5:-4]
      #   if af == "x8" :
      #     name += "+crop0"
      image = Image.open(p)
      data = np.asarray(image)
      a = data.reshape(from_shape, order='F')
      a = np.swapaxes(a, 1, 2)
      a = a.reshape(to_shape).astype('float32')
      b_X[j] = a
      mos = labels[name]['mos']
      # if model_struct == 'ALAS_DADS_CNN':
      #   b_y_mos[j] = mos
      #   b_y_spa[j] = np.array(labels[name]['spatial'])
      #   b_y_ang_gdd[j] = np.array(labels[name]['angular_gdd'])
      #   if not no_wlbp:
      #     b_y_ang_wlbp[j] = np.array(labels[name]['angular_wlbp']) 
      # else:
      b_y[j] = mos
        
        
    b_X=normalize(b_X)
    if batch_size>1:
      save_path=save_dir+f"tt_b{i}_bz{l}"
    else:
      # name = name.replace("train=", '').replace("test=",'')
      save_path=save_dir+name
    # if model_struct == 'ALAS_DADS_CNN':
    #   b_X={'lfi_input': b_X}
    #   b_y['mos'] = b_y_mos
    #   b_y['spatial'] = b_y_spa
    #   b_y['angular_gdd'] = b_y_ang_gdd
    #   if not no_wlbp:
    #     b_y['angular_wlbp'] = b_y_ang_wlbp
    #   b_dic = {'b_X':b_X, 'b_y':b_y}
    #   np.save(save_path,b_dic)
    #   np.save(save_path,b_dic)
    #   save_paths = glob.glob(save_dir+"*.npy", recursive=False)
    # else:
    np.savez(save_path,b_X=b_X, b_y=b_y)
    save_paths = glob.glob(save_dir+"*.npz", recursive=False)
      

    if verbose:
      t.lap()
      print(save_path,"was saved")
  print('success! -',save_dir,'were saved')
  if verbose:
    t.stop()
  return save_paths



def append_list_as_row(fn, l):
  with open(fn, 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(l)



def hasnan(X,model_struct):
  return False
  if model_struct!="ALAS_DADS_CNN":
    return np.any(np.isnan(X))
  for _,v in X.items():
    # print(np.isnan(np.sum(v)), np.any(np.isnan(v)))
    if np.any(np.isnan(v)):
      return True
  return False



def sqz_y_pred(y_pred):
  y_pred_sqzd={}
  y_pred_sqzd['mos'] = np.squeeze(y_pred[0]).tolist()
  y_pred_sqzd['spatial'] = np.squeeze(y_pred[1]).tolist()
  y_pred_sqzd['angular_gdd'] = np.squeeze(y_pred[2]).tolist()
  if not no_wlbp:
    y_pred_sqzd['angular_wlbp'] = np.squeeze(y_pred[3]).tolist()
  return y_pred_sqzd



def sqz_b_y(b_y):
  b_y_sqzd={}
  for k in b_y.keys():
    b_y_sqzd[k] = np.squeeze(b_y[k]).tolist()
  return b_y_sqzd



def scale_spatial(spatial):
  spa_min,spa_max = 25.811637432352,65.8772872416632
  return (spa_max-spatial)/(spa_max-spa_min)*4+1



def show_evaluation_results(loss_all, mae_all, mse_all, rmse, srcc, lcc):
  print('-'*10,'Evaluation Results','-'*10)
  print('-'*10,'loss: {:.4f} MAE: {:.4f} MSE: {:.4f}'.format(loss_all,mae_all,mse_all),'-'*10)
  print('+'*10,'RMSE: {:.4f} SRCC: {:.4f} LCC: {:.4f}'.format(rmse,srcc,lcc),'+'*10)
  return loss_all, mae_all, mse_all, rmse, srcc, lcc



def save_model_outputs(model_outputs,y_preds, y_trues, model_name, s, y_names):
  model_outputs_dir = get_version_dir()+"model_outputs"+s
  if not os.path.exists(model_outputs_dir):
    os.makedirs(model_outputs_dir)
  model_outputs_path=model_outputs_dir+"y="+model_name
  json.dump(model_outputs, open(model_outputs_path[:-3]+'.json', 'w'))
  if len(y_trues) == len(y_names):
    diffs = []
    for _,yp,yt in zip(y_names,y_preds,y_trues):
      diffs.append(yp-yt)
    data = {"y_name":y_names,"y_pred":y_preds,"y_true":y_trues,"diff":diffs}
    df = pd.DataFrame(data, columns=["y_name","y_pred","y_true","diff"])
    df.to_csv(model_outputs_path+".csv")
  return model_outputs_path



def normalize(X, norm_method="stand"):
  dataset_root, _, s = root_paths()
  if af=="x32":
    save_dir =dataset_root+ dn+"-tensor"+s
  else:
    save_dir =dataset_root+ dn+"-tensor-"+af+s
  if norm_method=="stand":
    save_path = save_dir+'mean_std.npz'
  else:
    save_path = save_dir+'min_max.npz'
  if not os.path.exists(save_path):
    print('The essential stats were not calculated!')
    return
  with np.load(save_path) as tr_stats:
    if norm_method=="stand":
      avg, std = tr_stats['avg'], tr_stats['std']
      normalized_X = (X-avg)/(std+1)
    else:
      min_a, max_a = tr_stats['min_a'], tr_stats['max_a']
      normalized_X = (X-min_a)/(max_a-min_a)
  return normalized_X



def get_version_paths(model_struct=None):
  print(get_version_dir(model_struct))
  _, _,s = root_paths()
  return glob.glob(get_version_dir(model_struct)+f'**{s}*.h5', recursive=True)



def get_version_dir(model_struct=None):
  _, model_root,s = root_paths()
  ver_dir = model_root+f"versions-{dn}{s}"
  if model_struct != None:
    ver_dir = model_root+f"versions-{dn}{s}{model_struct}{s}"
  if not os.path.exists(ver_dir):
    os.makedirs(ver_dir)
  return ver_dir



def root_paths():
  dataset_root = ".\Datasets\\" if sys.platform =='win32' else "./Datasets/"
  model_root = ".\\" if sys.platform =='win32' else "./"
  seperator = "\\" if sys.platform =='win32' else "/"
  return dataset_root, model_root, seperator



def kws_in_str(kws,st, strict):
  for kw in kws:
    if not kw in st and strict:
      return False
    if kw in st  and not strict:
      return True
  return True



def get_version_name(model_prex, ver_dir, file_type):
    n=0
    for i in range(1000):
      n=i
      file_name =model_prex +str(n)+file_type
      # keywords.append(file_type)
      versions_dirs = glob.glob(ver_dir+'*'+file_type)
      # n_used = len([d for d in versions_dirs if kws_in_str(keywords, d, True)] )>0
      # if not n_used:
      #   break
      if not file_name in versions_dirs:
        break
    model_save_dir=model_prex+str(n)
    return model_save_dir, n



def sort_batch_dirs(batch_dirs):
  return sorted(batch_dirs, key=lambda bn: int(bn[bn.find('_batch')+6:-4]))




if __name__ == "__main__":
  pass
