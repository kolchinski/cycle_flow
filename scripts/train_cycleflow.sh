set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout \
--gpu_ids -1 --display_id -1 --netG glow --invertible_G
