import os

os.system("python tools/train.py configs/r3det/r3det_r50_fpn_3x_hrsc_le90.py --work-dir results/hrsc/r3det/r50_gen_ISO2_new_train_finetune_3x --seed 42")
os.system("python tools/train.py configs/redet/redet_re50_refpn_3x_hrsc_le90_finetune.py --work-dir results/hrsc/redet/r50_gen_ISO2_new_train_finetune_3x --seed 42")
os.system("python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_3x_hrsc_le90_finetune.py --work-dir results/hrsc/oriented_rcnn/r50_gen_ISO2_new_train_finetune_3x --seed 42")
os.system("python tools/train.py configs/roi_trans/roi_trans_r50_fpn_3x_hrsc_le90_finetune.py --work-dir results/hrsc/roi_trans/r50_gen_ISO2_new_train_finetune_3x --seed 42")
os.system("python tools/train.py configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_3x_hrsc_rr_le90_finetune.py --work-dir results/hrsc/rotated_retinanet/r50_gen_ISO2_new_train_finetune_3x --seed 42")
os.system("python tools/train.py configs/s2anet/s2anet_r50_fpn_3x_hrsc_le90_finetune.py --work-dir results/hrsc/s2anet/r50_gen_ISO2_new_train_finetune_3x --seed 42")