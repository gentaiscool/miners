for gen_model in CohereForAI/aya-23-8B
do
    for model in sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0
    do
        python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit
        python icl.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit
        python icl.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit
        python icl.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit
        python icl.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit
done


python icl.py --dataset fire --src_lang tamil --cross --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1

python icl.py --dataset nusax --src_lang eng --cross --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1

python icl.py --dataset nollysenti --src_lang en --cross --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B --cuda --load_in_8bit --k 1




python icl.py --dataset fire --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0

python icl.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl.py --dataset nollysenti --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl.py --dataset lince_sa --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0







python icl1_1.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_2.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_3.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_4.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_5.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0

python icl1_1.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_2.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_3.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_4.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0
python icl1_5.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint sentence-transformers/LaBSE --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 0


python icl1_1.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_2.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_3.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_4.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_5.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1

python icl1_1.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_2.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_3.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_4.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_5.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint embed-multilingual-v3.0 --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1

python icl1_1.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_2.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_3.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_4.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_5.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1

python icl1_1.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_2.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_3.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_4.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1
python icl1_5.py --dataset sib200 --seed 42 --instruction "Generate a topic label for a given input." --model_checkpoint intfloat/multilingual-e5-large --gen_model_checkpoint CohereForAI/aya-23-8B  --cuda --load_in_8bit --k 1








