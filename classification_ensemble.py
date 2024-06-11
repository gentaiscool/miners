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

from collections import Counter

from utils import NusaXDataset, NusaTranslationDataset, TatoebaDataset, BUCCDataset, LinceMTDataset, PhincDataset, LinceSADataset, MassiveIntentDataset, Sib200Dataset, NollySentiDataset, MTOPIntentDataset, FIREDataset

from operator import add

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

def evaluate_classification(train_embeddings_all_models, test_embeddings_all_models, train_labels, k, weights):
    hyps = []

    for test_id in tqdm(range(len(test_embeddings_all_models[0]))):
        all_dists = []
        for model_checkpoint_id in range(len(train_embeddings_all_models)):
            train_embeddings = train_embeddings_all_models[model_checkpoint_id]
            test_embeddings = test_embeddings_all_models[model_checkpoint_id]
            
            dists = []
            batch_size = 128
            if len(train_embeddings) < batch_size:
                batch_size = len(test_embeddings) // 2
            
            num_of_batches = len(train_embeddings) // batch_size
    
            if (len(train_embeddings) % batch_size) > 0:
                num_of_batches += 1
    
            for i in range(num_of_batches):
                train_embedding = torch.FloatTensor(train_embeddings[i*batch_size:(i+1)*batch_size]).unsqueeze(1).cuda()
                
                test_embedding = torch.FloatTensor(test_embeddings[test_id]).unsqueeze(0)
                test_embedding = test_embedding.expand(len(train_embedding), -1).unsqueeze(1).cuda()
    
                # print(train_embedding.size(), test_embedding.size())
                
                dist = torch.cdist(test_embedding, train_embedding , p=2, compute_mode='use_mm_for_euclid_dist_if_necessary').squeeze().tolist()
    
                if isinstance(dist, float):
                    dist = [dist]
    
                for j in range(len(dist)):
                    dists.append([dist[j]* weights[model_checkpoint_id], train_labels[i*batch_size + j]])

            if len(all_dists) == 0:
                all_dists = dists
            else:
                for l in range(len(all_dists)):
                    all_dists[l][0] += dists[l][0]

        sorted_dists = sorted(all_dists,key=lambda l:l[0], reverse=False)[:k]
        all_indices = [obj[1] for obj in sorted_dists]
        c = Counter(all_indices)
        majority = c.most_common()[0][0]
        hyps.append(majority)
    return hyps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoints', type=str, nargs='+', required=True, help='a list of model checkpoints')
    parser.add_argument('--weights', type=float, nargs='+', required=True, help='a list of weights')
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--src_lang", type=str, default="x", help="source language")
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
    print("cross:", args.cross)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("###########################")

    set_seed(args.seed)

    if args.cross:
        output_dir = f"outputs/save_classification_cross_{args.src_lang}"
    else:
        output_dir = "outputs/save_classification"

    model_save_dir = ""
    for model_checkpoint_id in range(len(args.model_checkpoints)):
        model_checkpoint = args.model_checkpoints[model_checkpoint_id]
        if model_save_dir != "":
            model_save_dir += "_"
        model_save_dir += model_checkpoint.split("/")[-1] + "_" 
    model_save_dir += "_weights" + "_".join([str(w) for w in args.weights])
    
    
    models = []
    for model_checkpoint in args.model_checkpoints:
        print(">", model_checkpoint)
        if "embed-multilingual" in model_checkpoint:
            models.append(cohere.Client(COHERE_TOKEN))
            batch_size = 64
        elif "text-embedding-3-large" in model_checkpoint:
            models.append(OpenAI(api_key=OPENAI_TOKEN))
            batch_size = 64
        else:
            models.append(SentenceTransformer(model_checkpoint).cuda())
            batch_size = 64

    if args.dataset == "nusax":
        dataset = NusaXDataset(task="classification")
    if args.dataset == "lince_sa":
        dataset = LinceSADataset()
    if args.dataset == "massive_intent":
        dataset = MassiveIntentDataset()
    if args.dataset == "sib200":
        dataset = Sib200Dataset()
    if args.dataset == "nollysenti":
        dataset = NollySentiDataset(task="classification")
    if args.dataset == "mtop_intent":
        dataset = MTOPIntentDataset()
    if args.dataset == "fire":
        dataset = FIREDataset()

    for lang in dataset.LANGS:
        train_embeddings_all_models = []
        test_embeddings_all_models = []
        skip_it = False
        for model_id in range(len(args.model_checkpoints)):
            model_checkpoint = args.model_checkpoints[model_id]
            model = models[model_id]
            print(">", lang, model_checkpoint)
            train_embeddings = []
            
            if args.cross and lang == args.src_lang:
                print("skip src language eval", lang)
                skip_it = True
                break
                
            # get embeddings
            if args.cross:
                train_texts = dataset.train_data[args.src_lang]["source"]
                train_labels = dataset.train_data[args.src_lang]["target"]
            else:
                train_texts = dataset.train_data[lang]["source"]
                train_labels = dataset.train_data[lang]["target"]

            if "intfloat/multilingual-e5-large" in model_checkpoint:
                for train_text_id in range(len(train_texts)):
                    train_texts[train_text_id] = "query: " + train_texts[train_text_id]
    
            if len(train_texts) < batch_size:
                batch_size = len(train_texts) // 2
            num_of_batches = len(train_texts) // batch_size
    
            if (len(train_texts) % batch_size) > 0:
                num_of_batches += 1
    
            if args.cross:
                print("> train:", args.src_lang, num_of_batches)
            else:
                print("> train:", lang, num_of_batches)
            
            for i in tqdm(range(num_of_batches)):
                train_batch_text = train_texts[i*batch_size:(i+1)*batch_size]
                train_batch_label = train_labels[i*batch_size:(i+1)*batch_size]
    
                if "embed-multilingual" in model_checkpoint:
                    train_batch_embeddings = get_cohere_embedding(model, train_batch_text, model_checkpoint)
                elif "text-embedding-3-large" in model_checkpoint:
                    train_batch_embeddings = get_openai_embedding(model, train_batch_text, model_checkpoint)
                else:
                    train_batch_embeddings = model.encode(train_batch_text, normalize_embeddings=False)
    
                if len(train_embeddings) == 0:
                    train_embeddings = train_batch_embeddings
                else:
                    for emb in train_batch_embeddings:
                        train_embeddings = np.concatenate((train_embeddings, np.expand_dims(emb, axis=0)), axis=0) 
        
            # test
            test_texts = dataset.test_data[lang]["source"]
            test_labels = dataset.test_data[lang]["target"]

            if "intfloat/multilingual-e5-large" in model_checkpoint:
                for test_text_id in range(len(test_texts)):
                    test_texts[test_text_id] = "query: " + test_texts[test_text_id]
    
            if len(test_texts) < batch_size:
                batch_size = len(test_texts) // 2
            num_of_batches = len(test_texts) // batch_size
    
            if (len(test_texts) % batch_size) > 0:
                num_of_batches += 1
                
            print("> test:", lang, num_of_batches)
            test_embeddings = []
            for i in tqdm(range(num_of_batches)):
                test_batch_text = test_texts[i*batch_size:(i+1)*batch_size]
                test_batch_label = test_labels[i*batch_size:(i+1)*batch_size]
    
                if "embed-multilingual" in model_checkpoint:
                    test_batch_embeddings = get_cohere_embedding(model, test_batch_text, model_checkpoint)
                elif "text-embedding-3-large" in model_checkpoint:
                    test_batch_embeddings = get_openai_embedding(model, test_batch_text, amodel_checkpoint)
                else:
                    test_batch_embeddings = model.encode(test_batch_text, normalize_embeddings=False)
    
                if len(test_embeddings) == 0:
                    test_embeddings = test_batch_embeddings
                else:
                    for emb in test_batch_embeddings:
                        test_embeddings = np.concatenate((test_embeddings, np.expand_dims(emb, axis=0)), axis=0)
            
            train_embeddings_all_models.append(train_embeddings)
            test_embeddings_all_models.append(test_embeddings)
        
        if not os.path.exists(f"{output_dir}/{args.dataset}/{model_save_dir}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{model_save_dir}/seed_{args.seed}/")

        if skip_it:
            continue
            
        for k in [1,5,10]:
            key = lang
            # print(key, k, train_embeddings.shape, test_embeddings.shape)
            hyps = evaluate_classification(train_embeddings_all_models, test_embeddings_all_models, train_labels, k=k, weights=args.weights)
            obj = {}
            obj[f'acc'] = accuracy_score(test_labels, hyps)
            obj[f'prec'] = precision_score(test_labels, hyps, average="macro")
            obj[f'rec'] = recall_score(test_labels, hyps, average="macro")
            obj[f'f1'] = f1_score(test_labels, hyps, average="macro")
            print(obj)

            file_path = output_dir + "/" + args.dataset + "/" + model_save_dir + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)