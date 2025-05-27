# python -u main.py --num_heads=4 --num_layers=4 --batch_size=5000 --out_dir=out_temp_4

# python -u main.py --num_heads=4 --num_layers=4 --batch_size=100 --out_dir=out_batch_size_100

# python -u main.py --num_heads=4 --num_layers=4 --batch_size=1000 --out_dir=out_batch_size_1000

python -u main.py \
    --num_heads=4 --num_layers=4 \
    --batch_size=1000 \
    --train_min_chain_length=0 --train_max_chain_length=3 \
    --test_min_chain_length=0 --test_max_chain_length=3 \
    --test_ood_min_chain_length=4 --test_ood_max_chain_length=4 \
    --out_dir=out_batch_size_1000_chain_length_zero_to_three

