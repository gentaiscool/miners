for gen_model in bigscience/bloomz-3b bigscience/bloomz-1b7 bigscience/bloomz-560m
do
    python icl_ensemble.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
    python icl_ensemble.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
    python icl_ensemble.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
    python icl_ensemble.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
    python icl_ensemble.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
done




python icl_ensemble.py --dataset fire --src_lang tamil --cross --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7 --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset nollysenti --src_lang en --cross --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7 --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset nusax --src_lang eng --cross --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3


python icl_ensemble.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --gen_model_checkpoint bigscience/bloomz-1b7  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3




python icl_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --gen_model_checkpoint bigscience/bloomz-560m  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python icl_ensemble.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --gen_model_checkpoint bigscience/bloomz-560m  --cuda --load_in_8bit --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3








