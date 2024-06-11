python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint sentence-transformers/use-cmlm-multilingual
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-base --prompt "query: "
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-large --prompt "query: "
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-base
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-large
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint cis-lmu/glot500-base
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint microsoft/Multilingual-MiniLM-L12-H384
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint sentence-transformers/paraphrase-multilingual-mpnet-base-v2