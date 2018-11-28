set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout \
--display_id -1 --loadSize 128 --fineSize 64 --ngf 4 --ndf 8 --netG glow --invertible_G --gpu_ids -1 --use_affine_layers --print_freq 1
