python train.py \
  --dataroot /path/to/your/dataset \
  --name CDAGAN_Brats \
  --dataset_mode template \
  --model ssa_gan \
  --thresh 0.1 \
  --no_dropout \
  --no_flip \
  --input_nc 4 \
  --output_nc 4 \
  --use_se \
  --specnorm
