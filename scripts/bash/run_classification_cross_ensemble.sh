python classification_ensemble.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
python classification_ensemble.py --src_lang tamil --cross --dataset fire --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
python classification_ensemble.py --src_lang en --cross --dataset massive_intent --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75

python classification_ensemble.py --dataset nollysenti --src_lang en --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
python classification_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
python classification_ensemble.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python classification_ensemble.py --src_lang tamil --cross --dataset fire --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python classification_ensemble.py --src_lang en --cross --dataset massive_intent --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python classification_ensemble.py --dataset nollysenti --src_lang en --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3
python classification_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 3 3



python classification_ensemble.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
python classification_ensemble.py --src_lang tamil --cross --dataset fire --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
python classification_ensemble.py --src_lang en --cross --dataset massive_intent --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
python classification_ensemble.py --dataset nollysenti --src_lang en --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
python classification_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3

python classification_ensemble.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 1 1
python classification_ensemble.py --src_lang tamil --cross --dataset fire --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 1 1
python classification_ensemble.py --src_lang en --cross --dataset massive_intent --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 1 1
python classification_ensemble.py --dataset nollysenti --src_lang en --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 1 1
python classification_ensemble.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 1 1