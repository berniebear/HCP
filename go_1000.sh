export CUDA_VISIBLE_DEVICES=0
for i in {0..1000}
do
model=l1_hh32_reaky_adam_lr1e3_b2_${i}
echo $model
python main60.py --model_name $model --layers 2 --hidden 32 --bs 2 --seed $i
done

