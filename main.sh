attack_method=vnifgsm
model=resnet18
input_dir=./data
output_dir=./adv_data/$attack_method/$model

python main.py\
    --input_dir $input_dir \
    --output_dir $output_dir \
    --attack $attack_method \
    --model $model \
    --batchsize 32

python main.py\
    --input_dir $input_dir \
    --output_dir $output_dir \
    --attack $attack_method \
    --model $model \
    --eval