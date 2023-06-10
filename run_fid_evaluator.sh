# for baseline
python fid_evaluator.py \
    -p baseline/unet \
    --single \
    --dataset cifar \
    --data_dir <path to cifar> \
    --batch_size 256 \
    --sample_count 10000 \
    --img_size 32 \
    --use_ddim \
    --device cuda \
    -o <path to output>

# for double
python fid_evaluator.py \
    -p <path to pretrained> \
    --dataset cifar \
    --data_dir <path to cifar> \
    --batch_size 256 \
    --sample_count 10000 \
    --img_size 32 \
    --use_ddim \
    --ratio_type <that ratio type of the model> \
    --device cuda \
    -o <path to output>

