import os

os.system("python tools/test.py configs/r3det/r3det_r50_fpn_3x_hrsc_le90.py results/hrsc/r3det/r50_orig_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/r3det/r50_orig_train_finetune_3x --out results/hrsc/r3det/r50_orig_train_finetune_3x/one_img.pkl")
#os.system("python tools/test.py configs/redet/redet_re50_refpn_3x_hrsc_le90_finetune.py results/hrsc/redet/r50_gen_SSO_new_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/redet/r50_gen_SSO_new_train_finetune_3x --eval mAP")
#os.system("python tools/test.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_3x_hrsc_le90_finetune.py results/hrsc/oriented_rcnn/r50_gen_SSO_new_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/oriented_rcnn/r50_gen_SSO_new_train_finetune_3x --eval mAP")
#os.system("python tools/test.py configs/roi_trans/roi_trans_r50_fpn_3x_hrsc_le90_finetune.py results/hrsc/roi_trans/r50_gen_SSO_new_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/roi_trans/r50_gen_SSO_new_train_finetune_3x --eval mAP")
#os.system("python tools/test.py configs/rotated_fcos/rotated_fcos_r50_fpn_3x_hrsc_le90_finetune.py results/hrsc/rotated_fcos/r50_gen_SSO_new_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/rotated_fcos/r50_gen_SSO_new_train_finetune_3x --eval mAP")
#os.system("python tools/test.py configs/s2anet/s2anet_r50_fpn_3x_hrsc_le90_finetune.py results/hrsc/s2anet/r50_gen_SSO_new_train_finetune_3x/epoch_36.pth --work-dir results/hrsc/s2anet/r50_gen_SSO_new_train_finetune_3x --eval mAP")
