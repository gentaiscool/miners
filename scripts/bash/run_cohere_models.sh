python bitext.py --src_lang eng --dataset lince_mt --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python bitext.py --src_lang eng --dataset nusax --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
for i in de fr ru zh
do
    python bitext.py --src_lang $i --dataset bucc --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
done
python bitext.py --src_lang ind --dataset nusatranslation --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python bitext.py --src_lang eng --dataset phinc --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
for i in afr amh ang ara arq arz ast awa aze bel ben ber bos bre bul cat cbk ceb ces cha cmn cor csb cym dan deu dsb dtp ell epo est eus fao fin fra fry gla gle glg gsw heb hin hrv hsb hun hye ido ile ina ind isl ita jav jpn kab kat kaz khm kor kur kzj lat lfn lit lvs mal mar max mhr mkd mon nds nld nno nob nov oci orv pam pes pms pol por ron rus slk slv spa sqi srp swe swg swh tam tat tel tgl tha tuk tur tzl uig ukr urd uzb vie war wuu xho yid yue zsm
do
    python bitext.py --src_lang $i --dataset tatoeba --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
done
python bitext.py --src_lang en400 --dataset nollysenti --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset lince_sa --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset massive_intent --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset nusax --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset sib200 --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset nollysenti --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset mtop_intent --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset fire --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0

python classification.py --src_lang eng --cross --dataset nusax --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --dataset sib200 --src_lang eng_Latn --cross --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --src_lang tamil --cross --dataset fire --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0

python classification.py --dataset nollysenti --src_lang en --cross --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --src_lang en --cross --dataset massive_intent --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0
python classification.py --src_lang en --cross --dataset mtop_intent --seed 42 --cuda --model_checkpoint embed-multilingual-v3.0