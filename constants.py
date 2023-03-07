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


