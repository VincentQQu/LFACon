from PIL import Image, ImageChops
import numpy as np
from xtimer import Timer
import sys, glob, os, random
import pandas as pd
import utils, constants
# 773s before avg_std; after avg(479s) 1252s; after std(633s) 1885s;
t = Timer()
t.start()
exclude_ref = True
dataset_root, _, s = utils.root_paths()
dn = constants.dataset_name
af = constants.aug_factor ###-
img_format = constants.img_format



def flatten_dataset(save_dir):
  read_dir = dataset_root+dn+s
  read_img_paths = glob.glob(read_dir+"**/*."+img_format, recursive=True)
  ref_word = 'Reference'
  if dn=="SMART":
    ref_word='SRCs'
  for p in read_img_paths:
    if exclude_ref:
      if ref_word in p:
        continue
    parts = p.split(s)
    name = parts[-3]+'-'+parts[-2]+'-'+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  print('success! - flatten_dataset')
  if dn=="SMART":
    cut_black()
  t.lap()
  return save_dir



def trim():
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    bg = Image.new(img.mode, img.size, img.getpixel((w-1,h-1)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    # l,r = 0, 625*15
    # tp,b = 0, 434*15
    # window = (l,tp,r,b)
    window=bbox
    new_img = img.crop(window)
    new_img.save(read_dir+name,img_format)
    t.lap()
  print('success! - trim')
  t.lap()
  return read_dir



def cut_black():
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    l,r = 0, 625*15
    tp,b = 0, 434*15
    window = (l,tp,r,b)
    new_img = img.crop(window)
    new_img.save(read_dir+name,img_format)
    t.lap()
  print('success! - cut_black')
  t.lap()
  return read_dir



def resize(save_dir):
  read_dir = dataset_root+ dn+"-flatten" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    img = Image.open(p)
    w, h = img.size
    # left, top, right, bottom
    w_offset = int((w-constants.to_size[0])/2)
    h_offset = int((h-constants.to_size[1])/2)
    l,r = w_offset, w_offset+constants.to_size[0]
    tp,b = h_offset, h_offset+constants.to_size[1]
    window = (l,tp,r,b)
    new_img = img.crop(window)
    new_img.save(save_dir+name,img_format)
    t.lap()
  print('success! - resize')
  t.lap()
  return save_dir



def reduce_angular(save_dir):
  read_dir = dataset_root+ dn+"-resized" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  org_shape = constants.org_shape
  from_shape = constants.from_shape
  u_offset=int((org_shape[0]-from_shape[0])/2)
  v_offset=int((org_shape[2]-from_shape[2])/2)
  print(u_offset,v_offset)
  for p in read_img_paths:
    name = p.split(s)[-1]
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(org_shape, order='F')
    u_end, v_end =u_offset+from_shape[0], v_offset+from_shape[2]
    a = a[u_offset:u_end,:,v_offset:v_end,:,:]
    a = a.reshape(constants.new_size, order='F')
    new_img = Image.fromarray(a)
    new_img.save(save_dir+name,img_format)
    t.lap()
  print('success! - reduce_angular')
  t.lap()
  return save_dir



def rotate_flip(save_dir):
  read_dir = dataset_root+ dn+"-reduce-angular" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  rotate_angles = [0,90,180,270]
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    image = Image.open(p)
    for r in rotate_angles:
      img = image.rotate(r)
      new_name = name[:-4]+'+rotate'+str(r)+name[-4:]
      img.save(save_dir+new_name,img_format)
      img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
      new_name_f = new_name[:-4]+'+lrflip'+new_name[-4:]
      img_f.save(save_dir+new_name_f,img_format)
  print('success! - rotate_flip')
  t.lap()
  return save_dir



def crop(save_dir):
  read_dir = dataset_root+ dn+"-rotated-flipped" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  # 0, 1, 2, 3: 00,01,10,11
  h,w=constants.new_size[0],constants.new_size[1]
  # left, top, right, bottom
  crop_windows = [(0,0,int(w/2),int(h/2)), (int(w/2),0,w,int(h/2)), (0,int(h/2),int(w/2),h), (int(w/2),int(h/2),w,h)]
  for p in read_img_paths:
    name = p.split(s)[-1]
    print('processing',name,'...')
    image = Image.open(p)
    for i, cw in enumerate(crop_windows):
      img = image.crop(cw)
      new_name = name[:-4]+'+crop'+str(i)+name[-4:]
      img.save(save_dir+new_name,img_format)
  print('success! - crop')
  t.lap()
  return save_dir



def tr_tt_split(save_dir):
  if af =='x32':
    read_dir = dataset_root+ dn+"-crop" + s
  else:
    read_dir = dataset_root+ dn+"-rotated-flipped" + s
  read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
  random.seed(22) # 21
  random.shuffle(read_img_paths)
  assert constants.n_tr_tt == len(read_img_paths)
  split_bar = constants.n_tr
  tr_paths, tt_paths = read_img_paths[:split_bar], read_img_paths[split_bar:]
  for p in tr_paths:
    parts = p.split(s)
    name = 'train='+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  for p in tt_paths:
    parts = p.split(s)
    name = 'test='+parts[-1]
    print(save_dir+name)
    os.rename(p, save_dir+name)
  print('success! - tr_tt_split')
  t.lap()
  return save_dir



def calculate_tr_avg_std(save_dir):
  save_path = save_dir+'mean_std.npz'
  print(save_path)
  if os.path.exists(save_path):
    print('avg and std already calculated!')
    with np.load(save_path) as tr_ms:
      avg, std = tr_ms['avg'], tr_ms['std']
    return avg, std
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if af =='x32':
    read_dir = dataset_root+ dn+"-tr-tt" + s
  else:
    read_dir = dataset_root+ dn+"-tr-tt-"+af + s
  read_img_paths_tr = glob.glob(read_dir+"train=*."+img_format, recursive=False)
  print('calculating avg ...')
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  avg = std = np.zeros(to_shape,dtype=np.float32)
  n_tr = len(read_img_paths_tr)
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    avg += a.astype('float32')
  avg = avg/n_tr
  print(avg.shape)
  print(avg)
  np.savez(save_path,avg=avg, std=std)
  print('success! - avg')
  t.lap()
  print('calculating std ...')
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    std += np.square(a.astype('float32')-avg)
  std = np.sqrt(std/(n_tr-1))
  print(std.shape)
  print(std)
  print('success! - std')
  t.lap()
  np.savez(save_path,avg=avg, std=std)
  print('success! - avg and std are saved')
  t.lap()
  return avg, std



def calculate_tr_min_max(save_dir):
  save_path = save_dir+'min_max.npz'
  print(save_path)
  if os.path.exists(save_path):
    print('min and max already calculated!')
    with np.load(save_path) as tr_ms:
      min_a, max_a = tr_ms['min'], tr_ms['max']
    return min_a, max_a
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if af=="x32":
    read_dir = dataset_root+ dn+"-tr-tt" + s
  else:
    read_dir = dataset_root+ dn+"-tr-tt-"+af + s
  read_img_paths_tr = glob.glob(read_dir+"train=*."+img_format, recursive=False)
  print('calculating min ...')
  from_shape = constants.from_shape
  to_shape = constants.to_shape
  min_a = np.full(to_shape,255,dtype=np.uint8)
  max_a = np.full(to_shape,0,dtype=np.uint8)
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    min_a = np.stack((min_a,a)).min(axis=0)
  min_a = min_a.astype('float32')
  print(min_a.shape)
  print(min_a)
  np.savez(save_path,min_a=min_a, max_a=max_a)
  print('success! - min')
  t.lap()
  print('calculating max ...')
  for p in read_img_paths_tr:
    image = Image.open(p)
    data = np.asarray(image)
    a = data.reshape(from_shape, order='F')
    a = np.swapaxes(a, 1, 2)
    a = a.reshape(to_shape)
    max_a = np.stack((max_a,a)).max(axis=0)
  max_a = max_a.astype('float32')
  print(max_a.shape)
  print(max_a)
  print('success! - max')
  t.lap()
  np.savez(save_path,min_a=min_a, max_a=max_a)
  print('success! - min and max are saved')
  return min_a, max_a



if __name__ == "__main__":
  flatten_save_dir = dataset_root+ dn+"-flatten" + s
  if not os.path.exists(flatten_save_dir):
    os.makedirs(flatten_save_dir)
    flatten_dataset(flatten_save_dir)
    t.lap()
  resize_save_dir = dataset_root+ dn+"-resized" + s
  if not os.path.exists(resize_save_dir):
    os.makedirs(resize_save_dir)
    resize(resize_save_dir)
  reduce_angular_save_dir = dataset_root+ dn+"-reduce-angular" + s
  if not os.path.exists(reduce_angular_save_dir):
    os.makedirs(reduce_angular_save_dir)
    reduce_angular(reduce_angular_save_dir)
  rotate_flip_save_dir = dataset_root+ dn+"-rotated-flipped" + s
  if not os.path.exists(rotate_flip_save_dir):
    os.makedirs(rotate_flip_save_dir)
    rotate_flip(rotate_flip_save_dir)
  # if af == "x32":
  #   crop_save_dir = dataset_root+ dn+"-crop" + s
  #   if not os.path.exists(crop_save_dir):
  #     os.makedirs(crop_save_dir)
  #     crop(crop_save_dir)
  if af == "x32":
    tr_tt_save_dir = dataset_root+ dn+"-tr-tt" + s
  else:
    tr_tt_save_dir = dataset_root+ dn+"-tr-tt-"+ af + s
  if not os.path.exists(tr_tt_save_dir):
    os.makedirs(tr_tt_save_dir)
    tr_tt_split(tr_tt_save_dir)
  # exit()
  if af == "x32":
    tr_stats_save_dir = dataset_root+ dn+"-tensor" + s
  else:
    tr_stats_save_dir = dataset_root+ dn+"-tensor-"+ af + s
  if not os.path.exists(tr_stats_save_dir):
    os.makedirs(tr_stats_save_dir)
  calculate_tr_avg_std(tr_stats_save_dir)
  # calculate_tr_min_max(tr_stats_save_dir)
  # if af == "x32":
  #   tt_batch_save_dir = dataset_root+ dn+"-tensor" + s
  # else:
  #   tt_batch_save_dir = dataset_root+ dn+"-tensor-"+ af + s
  # if not os.path.exists(tt_batch_save_dir):
  #   os.makedirs(tt_batch_save_dir)
  # utils.generate_test_batches(1,model_struct='LFACon',save_dir=tt_batch_save_dir, normAL=True,verbose=True)





# def to_sais(save_dir):
#   read_dir = dataset_root+ dn+"-tr-tt" + s
#   read_img_paths = glob.glob(read_dir+"*."+img_format, recursive=False)
#   for p in read_img_paths:
#     name = p.split(s)[-1][:-4]
#     print('processing', name, '...')
#     image = Image.open(p)
#     data = np.asarray(image)
#     from_shape = (9, 512, 9, 512, 3)
#     a = data.reshape(from_shape, order='F')
#     a = np.swapaxes(a, 1, 2)
#     for i in range(from_shape[0]):
#       for j in range(from_shape[2]):
#         img = Image.fromarray(a[i][j][:][:][:])
#         d = save_dir+name+s
#         if not os.path.exists(d):
#           os.makedirs(d)
#         img.save(d+str(i)+'_'+str(j)+'.bmp',img_format)
#   print('success! - to_sais')
#   t.lap()
#   return save_dir