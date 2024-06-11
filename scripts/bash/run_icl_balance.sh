for k in 1 2 3
do
    python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint bigscience/bloomz-560m  --cuda --load_in_8bit --balance --k=$k
done