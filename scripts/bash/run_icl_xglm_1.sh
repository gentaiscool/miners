for gen_model in facebook/xglm-564M facebook/xglm-2.9B
do
    for model in intfloat/multilingual-e5-large sentence-transformers/LaBSE embed-multilingual-v3.0
    do
        python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
    done
done
