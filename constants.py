dataset_name = "MPI-LFA" # Win5-LID SMART
aug_factor = 'x8' # x32 x8
reduce_1 = True
no_wlbp=True
if dataset_name == "Win5-LID":
  # 9×9×434×625×3
  dataset_size=220
  to_size=(9*434, 9*434, 3)
  org_shape = (9,434,9,434,3)
  new_size = (7*434, 7*434, 3)
  new_shape = (7,434,7,434, 3)

elif dataset_name == "SMART":
  # 15×15×434×625×3
  dataset_size=256
  to_size=(15*434, 15*434, 3)
  org_shape = (15,434,15,434,3)
  new_size = (7*434, 7*434, 3)
  new_shape = (7,434,7,434, 3)

elif dataset_name == "MPI-LFA":
  dataset_size=336
  to_size=(15*434, 15*434, 3)
  org_shape = (15,434,15,434,3)
  new_size = (7*434, 7*434, 3)
  new_shape = (7,434,7,434, 3)



from_shape = (7,434,7,434,3)
to_shape=(7*7,434,434,3)
n_tr_tt = dataset_size*8 # 1408 : 5632

tr_tt_ratio = 0.2
n_tt = int(n_tr_tt*tr_tt_ratio) 
n_tr = n_tr_tt - n_tt
n_spa_feats =36
# gdd 1,8 wlbp 108,1
n_ang_feats_gdd = 8
n_ang_feats_wlbp = 108

img_format = "bmp" if dataset_name=="Win5-LID" else "png"
n_workers = 1
multip = False

# (6*(66.0+83.0+88.0+92.0+94.0+88.0+68.0+86.0+89.0+94.0+68.0+85.0+86.0+92.0+80.0+78.0+88.0+80.0+79.0+77.0+82.0+81.0+87.0+80.0+65.0+75.0+80.0+77.0+78.0+88.0+62.0+84.0+93.0)+3*(98.0+92.0+81.0+90.0)+2*(71.0+64.0+71.0) )/ (6*33+3*4+2*3)

# (6*(66.0+83.0+88.0+92.0+94.0+88.0+68.0+86.0+89.0+94.0+68.0+85.0+86.0+92.0+80.0+78.0+88.0+80.0+79.0+77.0+82.0+81.0+87.0+80.0+65.0+75.0+80.0+77.0+78.0+88.0+62.0+84.0+93.0+4*85.0)+3*(98.0+92.0+81.0+90.0)+2*(71.0+64.0+71.0) )/ (6*(33+4)+3*4+2*3)


# sorted(l, key=lambda dic: dic['a'])