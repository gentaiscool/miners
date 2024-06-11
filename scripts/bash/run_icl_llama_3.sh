for gen_model in meta-llama/Meta-Llama-3-8B-Instruct
do
    for model in intfloat/multilingual-e5-large sentence-transformers/LaBSE embed-multilingual-v3.0
    do
        python icl.py --dataset fire --src_lang tamil --cross --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset nollysenti --src_lang en --cross --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model --cuda --load_in_8bit --k 1
        python icl.py --dataset nusax --src_lang eng --cross --seed 42 --instruction "Generate a sentiment label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
        python icl.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input.\nPlease only output the label." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1
    done
done