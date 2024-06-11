python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint sentence-transformers/use-cmlm-multilingual
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-base --prompt "query: "
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-large --prompt "query: "
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-base
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-large
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint cis-lmu/glot500-base
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint microsoft/Multilingual-MiniLM-L12-H384
python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint sentence-transformers/paraphrase-multilingual-mpnet-base-v2