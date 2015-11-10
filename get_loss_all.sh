#!/bin/bash
#python get_loss.py output/short_same_class5_2d_lr5.out
#python get_loss.py output/short_same_class10_2d_lr5.out
#python get_loss.py output/short_same_class20_2d_lr5.out

#python get_loss.py output/short_same_class5_3d_lr5.out
#python get_loss.py output/short_same_class10_3d_lr5_cont.out
#python get_loss.py output/short_same_class20_3d_lr5_cont.out

python get_loss.py output/short_same_class20_3d_lr3.out
python get_loss.py output/short_same_class20_3d_lr4.out
python get_loss.py output/short_same_class20_3d_lr5.out

python get_loss.py output/short_same_class74_3d_lr3.out
python get_loss.py output/short_same_class74_3d_lr4.out
python get_loss.py output/short_same_class74_3d_lr5.out
