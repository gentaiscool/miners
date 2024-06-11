for gen_model in bigscience/bloomz-3b bigscience/bloomz-1b7 bigscience/bloomz-560m CohereForAI/aya-23-8B gpt-3.5-turbo-0125 gpt-4o-2024-05-13 meta-llama/Meta-Llama-3-8B-Instruct
do
    for model in intfloat/multilingual-e5-large
    do
        for min_percentile in 0 10 20 30 40 50 60 70 80 90 99
        do
            python icl_percentile.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1 --sample_percentile --min_percentile $min_percentile --max_percentile 100
        done
    done 
done

for gen_model in bigscience/bloomz-3b bigscience/bloomz-1b7 bigscience/bloomz-560m CohereForAI/aya-23-8B gpt-3.5-turbo-0125 gpt-4o-2024-05-13 meta-llama/Meta-Llama-3-8B-Instruct
do
    for model in intfloat/multilingual-e5-large
    do
        for min_percentile in 0 10 20 30 40 50 60 70 80 90 99
        do
            python icl_percentile.py --dataset nusax --seed 42 --instruction "Generate a sentiment label for a given input." --model_checkpoint $model --gen_model_checkpoint $gen_model  --cuda --load_in_8bit --k 1 --sample_percentile --min_percentile $min_percentile --max_percentile 100
        done
    done 
done
