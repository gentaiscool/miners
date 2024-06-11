import os
import random
import numpy as np
import argparse
import json
from tabulate import tabulate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="number of k")
    args = parser.parse_args()

    print("###########################")
    print("k:", args.k)
    print("###########################")

    datasets = ["massive_intent","nollysenti","nusax","sib200","fire","lince_sa"]
    metrics = ["acc","acc","f1","acc","acc","acc"]
    models = ["sentence-transformers/LaBSE", "sentence-transformers/use-cmlm-multilingual", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "microsoft/Multilingual-MiniLM-L12-H384", "cis-lmu/glot500-base", "FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-large", "embed-multilingual-v3.0", "text-embedding-3-large","LaBSE__multilingual-e5-large__weights0.25_0.75", "LaBSE__multilingual-e5-large__embed-multilingual-v3.0__weights1.0_2.0_3.0"]
    total_langs = [51, 5, 12, 205, 2, 1]
    tasks = ["mono","mono","mono","mono","cs","cs"]
    all_tasks = ["mono","cs"]
    
    all_res = {}
    headers = ["models"] + datasets + ["avg"]
    rows = []

    tasks_headers = ["models", "mono", "cs", "avg"]
    tasks_rows = []
    for model_name in models:
        row = [model_name.split("/")[-1]]
        tasks_row = [model_name.split("/")[-1]]
        
        all_nums = []
        all_nums_per_task = {}
        for i in range(len(datasets)):
            task = tasks[i]
            if task not in all_nums_per_task:
                all_nums_per_task[task] = []
            dataset = datasets[i]
            output_dir = "logs/save_classification" + "/" + dataset + "/" + model_name + "/seed_42/"
            
            try:
                files_list = os.listdir(output_dir)
            
                res = []
                # print(files_list)
                for f in files_list:
                    if ".json" not in f:
                        continue
                    f = f.replace(".json","")
                    language_k = "_".join(f.split("_")[1:-1])
                    file_k = int(f.split("_")[-1])
                    if file_k == args.k:
                        # print(language_k, file_k)
                        json_obj = json.load(open(output_dir + "/" + f + ".json"))
                        res.append(json_obj[metrics[i]])
    
                incomplete = False
                if total_langs[i] != len(res):
                    print(">>>>>>>", dataset, model_name, "experiment is not completed, missing:", total_langs[i]-len(res))
                    incomplete = True
                
                all_res[model_name + "_" + dataset] = np.mean(np.array(res))
                if incomplete:
                    row.append(str(round(float(np.mean(np.array(res)) * 100), 2)) + "(IC)")
                else:
                    row.append(str(round(float(np.mean(np.array(res)) * 100), 2)))
                all_nums.append(round(float(np.mean(np.array(res)) * 100), 2))
                all_nums_per_task[task].append(round(float(np.mean(np.array(res)) * 100), 2))
            except:
                print(output_dir, "is missing")
                row.append("N/A")
        if len(all_nums) == 0:
            row.append("N/A")
        else:
            row.append(round(float(np.mean(np.array(all_nums))), 2))
            for task in all_tasks:
                all_nums_per_task[task] = (round(float(np.mean(np.array(all_nums_per_task[task]))), 2))
                tasks_row.append(round(float(np.mean(np.array(all_nums_per_task[task]))), 2))
            tasks_row.append(np.mean(tasks_row[1:]))
        # print(row)
        rows.append(row)
        tasks_rows.append(tasks_row)
        # print(model_name, all_nums_per_task)
    # print(all_res)
    print(tabulate(rows, headers, tablefmt="plain"))

    print(tabulate(tasks_rows, tasks_headers, tablefmt="plain"))
                