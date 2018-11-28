set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout \
--display_id -1 --loadSize 128 --fineSize 64 --netG glow --invertible_G --use_affine_layers
