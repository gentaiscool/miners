import os
import torch
import random
import numpy as np
import argparse
import json
import cohere
from openai import OpenAI

from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from utils import NusaXDataset, NusaTranslationDataset, TatoebaDataset, BUCCDataset, LinceMTDataset, PhincDataset, NollySentiDataset

OPENAI_TOKEN = ""
COHERE_TOKEN = ""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_openai_embedding(model, texts, checkpoint="text-embedding-3-large"):
    data = model.embeddings.create(input = texts, model=checkpoint).data
    embeddings = []
    for obj in data:
        embeddings.append(obj.embedding)
    return embeddings
    
def get_cohere_embedding(model, texts, model_checkpoint):
    response = model.embed(texts=texts, model=model_checkpoint, input_type="search_query")
    return response.embeddings

def evaluate_bitext_mining(source_embeddings_all_models, target_embeddings_all_models, k, weights):
    hyps = []
    golds = []

    all_dists_models = []
    for model_id in range(len(source_embeddings_all_models)):
        source_embeddings = source_embeddings_all_models[model_id]
        target_embeddings = target_embeddings_all_models[model_id]

        all_dists_samples = []
        for source_id in tqdm(range(len(source_embeddings))):
            dists = []
            batch_size = 128
            if len(target_embeddings) < batch_size:
                batch_size = len(target_embeddings) // 2
            
            num_of_batches = len(target_embeddings) // batch_size
    
            if (len(target_embeddings) % batch_size) > 0:
                num_of_batches += 1
    
            for i in range(num_of_batches):
                target_embedding = torch.FloatTensor(target_embeddings[i*batch_size:(i+1)*batch_size]).unsqueeze(1).cuda()
                
                source_embedding = torch.FloatTensor(source_embeddings[source_id]).unsqueeze(0)
                source_embedding = source_embedding.expand(len(target_embedding), -1).unsqueeze(1).cuda()
                
                dist = torch.cdist(source_embedding, target_embedding, p=2, compute_mode='use_mm_for_euclid_dist_if_necessary').squeeze().tolist()
    
                if isinstance(dist, float):
                    dist = [dist]
    
                for j in range(len(dist)):
                    dists.append([dist[j]* weights[model_id], i*batch_size + j])

            all_dists_samples.append(dists)
        all_dists_models.append(all_dists_samples)

    all_final_dists = []
    for sample_id in range(len(all_dists_models[0])):
        temp_dists = []
        for model_id in range(len(all_dists_models)):
            dists = all_dists_models[model_id][sample_id]
            if len(temp_dists) == 0:
                temp_dists = dists
            else:
                for obj_id in range(len(dists)):
                    temp_dists[obj_id][0] += dists[obj_id][0]
        all_final_dists.append(temp_dists)
        
    for sample_id in range(len(all_final_dists)):
        dists = all_final_dists[sample_id]
        
        sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)[:k]
        all_indices = [obj[1] for obj in sorted_dists]

        if sample_id in all_indices:
            hyps.append(sample_id)
        else:
            hyps.append(all_indices[0])
        golds.append(sample_id)
    return hyps, golds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoints', type=str, nargs='+', help='a list of model checkpoints')
    parser.add_argument('--weights', type=float, nargs='+', required=True, help='a list of weights')
    parser.add_argument("--src_lang", type=str, default="eng", help="source language")
    parser.add_argument("--dataset", type=str, default="mtop", help="snips or mtop or multi-nlu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    print("src_lang:", args.src_lang)
    print("dataset:", args.dataset)
    print("model_checkpoints:", args.model_checkpoints)
    print("weights:", args.weights)
    print("seed:", args.seed)
    print("cuda:", args.cuda)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("###########################")

    set_seed(args.seed)

    output_dir = "outputs/save_bitext"

    model_save_dir = ""
    for model_checkpoint in args.model_checkpoints:
        if model_save_dir != "":
            model_save_dir += "_"
        model_save_dir += model_checkpoint.split("/")[-1] + "_" 
    model_save_dir += "_weights" + "_".join([str(w) for w in args.weights])

    models = []
    for model_checkpoint in args.model_checkpoints:
        if "embed-multilingual" in model_checkpoint:
            models.append(cohere.Client(COHERE_TOKEN))
            batch_size = 64
        elif "text-embedding-3-large" in model_checkpoint:
            models.append(OpenAI(api_key=OPENAI_TOKEN))
            batch_size = 64
        else:
            models.append(SentenceTransformer(model_checkpoint).cuda())
            batch_size = 128
    
    if args.dataset == "nusax":
        dataset = NusaXDataset(task="bitext")
    if args.dataset == "nusatranslation":
        dataset = NusaTranslationDataset(src_lang=args.src_lang)
    if args.dataset == "tatoeba":
        dataset = TatoebaDataset(src_lang=args.src_lang)
    if args.dataset == "bucc":
        dataset = BUCCDataset(src_lang=args.src_lang)
    if args.dataset == "lince_mt":
        dataset = LinceMTDataset(src_lang=args.src_lang)
    if args.dataset == "phinc":
        dataset = PhincDataset(src_lang=args.src_lang)
    if args.dataset == "nollysenti":
        dataset = NollySentiDataset(src_lang=args.src_lang)

    print(">", dataset.LANGS)
    target_embeddings_all_models = {}
    for target_lang in dataset.LANGS:
        for model_id in range(len(args.model_checkpoints)):
            model_checkpoint = args.model_checkpoints[model_id]
            model = models[model_id]
            
            source_embeddings = []
            target_embeddings = {}
            
            # get embeddings
            key = args.src_lang + "_" + target_lang
            if target_lang != args.src_lang:
                target_embeddings = {"source":[], "target":[]}
            else:
                continue
    
            if len(dataset.train_data[key]["target"]) < batch_size:
                batch_size = len(dataset.train_data[key]["target"]) // 2
                
            num_of_batches = len(dataset.train_data[key]["target"]) // batch_size
    
            if (len(dataset.train_data[key]) % batch_size) > 0:
                num_of_batches += 1
                
            print(key, target_lang, num_of_batches)
            
            for i in tqdm(range(num_of_batches)):
                source_batch_data = dataset.train_data[key]["source"][i*batch_size:(i+1)*batch_size]
                target_batch_data = dataset.train_data[key]["target"][i*batch_size:(i+1)*batch_size]

                if "intfloat/multilingual-e5" in model_checkpoint:
                    for data_id in range(len(source_batch_data)):
                        source_batch_data[data_id] = "query: " + source_batch_data[data_id]
                    for data_id in range(len(target_batch_data)):
                        target_batch_data[data_id] = "query: " + target_batch_data[data_id]
                
                if "embed-multilingual" in model_checkpoint:
                    source_batch_embeddings = get_cohere_embedding(model, source_batch_data, model_checkpoint)
                    target_batch_embeddings = get_cohere_embedding(model, target_batch_data, model_checkpoint)
                elif "text-embedding-3-large" in model_checkpoint:
                    source_batch_embeddings = get_openai_embedding(model, source_batch_data, model_checkpoint)
                    target_batch_embeddings = get_openai_embedding(model, target_batch_data, model_checkpoint)
                else:
                    source_batch_embeddings = model.encode(source_batch_data, normalize_embeddings=False)
                    target_batch_embeddings = model.encode(target_batch_data, normalize_embeddings=False)
                
                if len(target_embeddings["source"]) == 0:
                    target_embeddings["source"] = source_batch_embeddings
                else:
                    for emb in source_batch_embeddings:
                        target_embeddings["source"] = np.concatenate((target_embeddings["source"], np.expand_dims(emb, axis=0)), axis=0)
                
                if len(target_embeddings["target"]) == 0:
                    target_embeddings["target"] = target_batch_embeddings
                else:
                    for emb in target_batch_embeddings:
                        target_embeddings["target"] = np.concatenate((target_embeddings["target"], np.expand_dims(emb, axis=0)), axis=0)

            if key not in target_embeddings_all_models:
                target_embeddings_all_models[key] = []
            target_embeddings_all_models[key].append(target_embeddings)
    
    if not os.path.exists(f"{output_dir}/{args.dataset}/{model_save_dir}/seed_{args.seed}/"):
        os.makedirs(f"{output_dir}/{args.dataset}/{model_save_dir}/seed_{args.seed}/")

    for k in [1,5,10]:
        print(">>>>>>>>>>", target_embeddings_all_models.keys())
        for key in target_embeddings_all_models:
            print(">", key)

            source_emb_all_models = []
            target_emb_all_models = []

            for model_checkpoint_id in range(len(target_embeddings_all_models[key])):   
                # print(">>>", target_embeddings_all_models[key][model_checkpoint_id].keys())
                source_emb = target_embeddings_all_models[key][model_checkpoint_id]["source"]
                target_emb = target_embeddings_all_models[key][model_checkpoint_id]["target"]
                source_emb_all_models.append(source_emb)
                target_emb_all_models.append(target_emb)
            
            print(k, len(source_emb_all_models), len(target_emb_all_models))
            hyps, golds = evaluate_bitext_mining(source_emb_all_models, target_emb_all_models, k=k, weights=args.weights)
            
            obj = {}
            obj[f'acc'] = accuracy_score(golds, hyps)
            obj[f'prec'] = precision_score(golds, hyps, zero_division=0.0, average="weighted")
            obj[f'rec'] = recall_score(golds, hyps, zero_division=0.0, average="weighted")
            obj[f'f1'] = f1_score(golds, hyps, zero_division=0.0, average="weighted")
            print(obj)

            file_path = output_dir + "/" + args.dataset + f"/{model_save_dir}/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)