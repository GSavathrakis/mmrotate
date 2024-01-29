import os 


for i in range(3,73):
	os.system(f'python tools/test.py configs/r3det/r3det_r101_fpn_6x_hrsc_le90.py results/hrsc/r3det/multi_class/re101_72epochs/epoch_{i}.pth --work-dir results/hrsc/r3det/multi_class/re101_72epochs/test --eval mAP')