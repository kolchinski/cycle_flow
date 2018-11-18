set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout \
--display_id -1 --fineSize 64 --ngf 4 --ndf 4 --netG glow --invertible_G --gpu_ids -1
