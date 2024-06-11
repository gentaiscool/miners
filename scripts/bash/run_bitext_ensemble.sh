CUDA_VISIBLE_DEVICES=0 python bitext_ensemble.py --src_lang eng --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
CUDA_VISIBLE_DEVICES=1 python bitext_ensemble.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
for i in de fr ru zh
do
    (run on h100) python bitext_ensemble.py --src_lang $i --dataset bucc --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
done

CUDA_VISIBLE_DEVICES=3 python bitext_ensemble.py --src_lang en400 --dataset nollysenti --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75

(run on h100) python bitext_ensemble.py --src_lang eng --dataset phinc --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75

CUDA_VISIBLE_DEVICES=5 python bitext_ensemble.py --src_lang ind --dataset nusatranslation --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75

for i in afr amh ang ara arq arz ast awa aze bel ben ber bos bre bul cat cbk ceb ces cha cmn cor csb cym dan deu dsb dtp ell epo est eus fao fin fra fry gla gle glg gsw heb hin hrv hsb hun hye ido ile ina ind isl ita jav jpn kab kat kaz khm kor kur kzj lat lfn lit lvs mal mar max mhr mkd mon nds nld nno nob nov oci orv pam pes pms pol por ron rus slk slv spa sqi srp swe swg swh tam tat tel tgl tha tuk tur tzl uig ukr urd uzb vie war wuu xho yid yue zsm
do
    CUDA_VISIBLE_DEVICES=6 python bitext_ensemble.py --src_lang $i --dataset tatoeba --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large --weights 0.25 0.75
done



CUDA_VISIBLE_DEVICES=7 python bitext_ensemble.py --src_lang eng --dataset nusax --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
CUDA_VISIBLE_DEVICES=3 python bitext_ensemble.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3

for i in de fr ru zh
do
    (run on h100) python bitext_ensemble.py --src_lang $i --dataset bucc --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
done

CUDA_VISIBLE_DEVICES=2 python bitext_ensemble.py --src_lang en400 --dataset nollysenti --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3

(run on h100) python bitext_ensemble.py --src_lang eng --dataset phinc --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3

CUDA_VISIBLE_DEVICES=4 python bitext_ensemble.py --src_lang ind --dataset nusatranslation --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3


for i in afr amh ang ara arq arz ast awa aze bel ben ber bos bre bul cat cbk ceb ces cha cmn cor csb cym dan deu dsb dtp ell epo est eus fao fin fra fry gla gle glg gsw heb hin hrv hsb hun hye ido ile ina ind isl ita jav jpn kab kat kaz khm kor kur kzj lat lfn lit lvs mal mar max mhr mkd mon nds nld nno nob nov oci orv pam pes pms pol por ron rus slk slv spa sqi srp swe swg swh tam tat tel tgl tha tuk tur tzl uig ukr urd uzb vie war wuu xho yid yue zsm
do
    (run on h100) python bitext_ensemble.py --src_lang $i --dataset tatoeba --seed 42 --cuda  --model_checkpoints sentence-transformers/LaBSE intfloat/multilingual-e5-large embed-multilingual-v3.0 --weights 1 2 3
done