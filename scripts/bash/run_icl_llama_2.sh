for gen_model in meta-llama/Meta-Llama-3-8B-Instruct
do
    for model in sentence-transformers/LaBSE
    do
        python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 0
        python icl.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 0
        python icl.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 0
        python icl.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 0
        python icl.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 0
    done
done