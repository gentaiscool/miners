for i in afr amh ang ara arq arz ast awa aze bel ben ber bos bre bul cat cbk ceb ces cha cmn cor csb cym dan deu dsb dtp ell epo est eus fao fin fra fry gla gle glg gsw heb hin hrv hsb hun hye ido ile ina ind isl ita jav jpn kab kat kaz khm kor kur kzj lat lfn lit lvs mal mar max mhr mkd mon nds nld nno nob nov oci orv pam pes pms pol por ron rus slk slv spa sqi srp swe swg swh tam tat tel tgl tha tuk tur tzl uig ukr urd uzb vie war wuu xho yid yue zsm
do
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint sentence-transformers/LaBSE
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint sentence-transformers/use-cmlm-multilingual
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-base --prompt "query: "
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint intfloat/multilingual-e5-large --prompt "query: "
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-base
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint FacebookAI/xlm-roberta-large
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint cis-lmu/glot500-base
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint microsoft/Multilingual-MiniLM-L12-H384
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint sentence-transformers/paraphrase-multilingual-mpnet-base-v2
done