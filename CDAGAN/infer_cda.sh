set -ex
python test.py \
--dataroot /path/to/your/dataset \
--name CDAGAN_Brats \
--model ssa_gan \
--phase test \
--num_test 50000 \
--no_dropout \
--thresh 0.1 \
--no_flip \
--input_nc 4 \
--output_nc 4 \
--results_dir results/ssagan_atob \
--test_direction AtoB \
--use_se \
--specnorm
