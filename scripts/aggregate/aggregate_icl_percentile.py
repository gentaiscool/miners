import os
import random
import numpy as np
import argparse
import json
from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="number of k")
    args = parser.parse_args()

    print("###########################")
    print("k:", args.k)
    print("###########################")

    percentiles = [0,10,20,30,40,50,60,70,80,90,99]
    metrics = ["f1"] * 11
    models = ["sentence-transformers/LaBSE","intfloat/multilingual-e5-large", "embed-multilingual-v3.0"]
    gen_models = ["bigscience/bloomz-560m"]
    # gen_models = ["bigscience/bloomz-560m", "bigscience/bloomz-1b7", "bigscience/bloomz-3b", "CohereForAI/aya-23-8B", "gpt-3.5-turbo-0125", "gpt-4o-2024-05-13"]
    total_langs = [12] * 11

    
    for gen_model_name in gen_models:
        print("#"*20)
        print(f"gen model:{gen_model_name}")
        print("#"*20)

        all_res = {}
        headers = ["models"] + percentiles + ["avg"]
        rows = []
    
        for model_name in models:
            row = [model_name.split("/")[-1]]
            tasks_row = [model_name.split("/")[-1]]
            
            all_nums = []
            all_nums_per_task = {}
            for i in range(len(percentiles)):
                percentile = str(percentiles[i])
                output_dir = "logs/save_icl_8bit_percentile" + f"/{percentile}_100/nusax/" + gen_model_name + "/" + model_name + "/seed_42/"
                
                try:
                    files_list = os.listdir(output_dir)
                
                    res = []
                    # print(files_list)
                    for f in files_list:
                        if ".json" not in f or "pred" in f or "prompt" in f:
                            continue
                        f = f.replace(".json","")
                        language_k = "_".join(f.split("_")[1:-1])
                        file_k = int(f.split("_")[-1])
                        # print(">>>")
                        if file_k == args.k:
                            # print(language_k, file_k)
                            json_obj = json.load(open(output_dir + "/" + f + ".json"))
                            res.append(json_obj[metrics[i]])
                            
                    incomplete = False
                    if total_langs[i] != len(res):
                        print(">>>>>>>", percentile, model_name, "experiment is not completed, missing:", total_langs[i]-len(res))
                        incomplete = True

                    all_res[model_name + "_" + percentile] = np.mean(np.array(res))
                    if incomplete:
                        row.append(str(round(float(np.mean(np.array(res)) * 100), 2)) + "(IC)")
                    else:
                        row.append(str(round(float(np.mean(np.array(res)) * 100), 2)))
                    all_nums.append(round(float(np.mean(np.array(res)) * 100), 2))
                except:
                    print(output_dir, "is missing")
                    row.append("N/A")
            if len(all_nums) == 0:
                row.append("N/A")
            else:
                row.append(round(float(np.mean(np.array(all_nums))), 2))

            rows.append(row)
        print(headers)
        print(tabulate(rows, headers, tablefmt="plain"))
        print("\n")               