import os
import torch
import random
import numpy as np
import argparse
import json
import cohere
from openai import OpenAI

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

def evaluate_bitext_mining(source_embeddings, target_embeddings, k):
    hyps = []
    golds = []
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
                dists.append([dist[j], i*batch_size + j])

        sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)[:k]
        all_indices = [obj[1] for obj in sorted_dists]

        if source_id in all_indices:
            hyps.append(source_id)
        else:
            hyps.append(all_indices[0])
        golds.append(source_id)
    return hyps, golds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model")
    parser.add_argument("--src_lang", type=str, default="eng", help="source language")
    parser.add_argument("--dataset", type=str, default="mtop", help="snips or mtop or multi-nlu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    print("src_lang:", args.src_lang)
    print("dataset:", args.dataset)
    print("model_checkpoint:", args.model_checkpoint)
    print("seed:", args.seed)
    print("cuda:", args.cuda)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("prompt:", args.prompt)
    print("###########################")

    set_seed(args.seed)

    output_dir = "outputs/save_bitext"

    if "embed-multilingual" in args.model_checkpoint:
        model = cohere.Client(COHERE_TOKEN)
        batch_size = 64
    elif "text-embedding-3-large" in args.model_checkpoint:
        model = OpenAI(api_key=OPENAI_TOKEN)
        batch_size = 64
    else:
        model = SentenceTransformer(args.model_checkpoint).cuda()
        batch_size = 128

    source_embeddings = []
    target_embeddings = {}

    if args.dataset == "nusax":
        dataset = NusaXDataset(prompt=args.prompt)
    if args.dataset == "nusatranslation":
        dataset = NusaTranslationDataset(prompt=args.prompt, src_lang=args.src_lang)
    if args.dataset == "tatoeba":
        dataset = TatoebaDataset(prompt=args.prompt, src_lang=args.src_lang)
    if args.dataset == "bucc":
        dataset = BUCCDataset(prompt=args.prompt, src_lang=args.src_lang)
    if args.dataset == "lince_mt":
        dataset = LinceMTDataset(prompt=args.prompt, src_lang=args.src_lang)
    if args.dataset == "phinc":
        dataset = PhincDataset(prompt=args.prompt, src_lang=args.src_lang)
    if args.dataset == "nollysenti":
        dataset = NollySentiDataset(prompt=args.prompt, src_lang=args.src_lang)

    for target_lang in dataset.LANGS:
        # get embeddings
        key = args.src_lang + "_" + target_lang
        if target_lang != args.src_lang:
            target_embeddings[key] = {"source":[], "target":[]}
        else:
            continue

        if len(dataset.train_data[key]["target"]) < batch_size:
            batch_size = len(dataset.train_data[key]["target"]) // 2
            
        num_of_batches = len(dataset.train_data[key]["target"]) // batch_size

        if (len(dataset.train_data[key]) % batch_size) > 0:
            num_of_batches += 1
            
        print(target_lang, num_of_batches)
        
        for i in tqdm(range(num_of_batches)):
            source_batch_data = dataset.train_data[key]["source"][i*batch_size:(i+1)*batch_size]
            target_batch_data = dataset.train_data[key]["target"][i*batch_size:(i+1)*batch_size]
            if "embed-multilingual" in args.model_checkpoint:
                source_batch_embeddings = get_cohere_embedding(model, source_batch_data, args.model_checkpoint)
                target_batch_embeddings = get_cohere_embedding(model, target_batch_data, args.model_checkpoint)
            elif "text-embedding-3-large" in args.model_checkpoint:
                source_batch_embeddings = get_openai_embedding(model, source_batch_data, args.model_checkpoint)
                target_batch_embeddings = get_openai_embedding(model, target_batch_data, args.model_checkpoint)
            else:
                source_batch_embeddings = model.encode(source_batch_data, normalize_embeddings=False)
                target_batch_embeddings = model.encode(target_batch_data, normalize_embeddings=False)
            
            if len(target_embeddings[key]["source"]) == 0:
                target_embeddings[key]["source"] = source_batch_embeddings
            else:
                for emb in source_batch_embeddings:
                    target_embeddings[key]["source"] = np.concatenate((target_embeddings[key]["source"], np.expand_dims(emb, axis=0)), axis=0)
            
            if len(target_embeddings[key]["target"]) == 0:
                target_embeddings[key]["target"] = target_batch_embeddings
            else:
                for emb in target_batch_embeddings:
                    target_embeddings[key]["target"] = np.concatenate((target_embeddings[key]["target"], np.expand_dims(emb, axis=0)), axis=0)

        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.model_checkpoint}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.model_checkpoint}/seed_{args.seed}/")

    for k in [1,5,10]:
        for key in target_embeddings:
            print(">", key)
            source_emb = target_embeddings[key]["source"]
            target_emb = target_embeddings[key]["target"]
            print(k, source_emb.shape, target_emb.shape)
            hyps, golds = evaluate_bitext_mining(source_emb, target_emb, k=k)
            
            obj = {}
            obj[f'acc'] = accuracy_score(golds, hyps)
            obj[f'prec'] = precision_score(golds, hyps, zero_division=0.0, average="weighted")
            obj[f'rec'] = recall_score(golds, hyps, zero_division=0.0, average="weighted")
            obj[f'f1'] = f1_score(golds, hyps, zero_division=0.0, average="weighted")
            print(obj)

            file_path = output_dir + "/" + args.dataset + "/" + args.model_checkpoint + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)