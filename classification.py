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

def evaluate_classification(train_embeddings, test_embeddings, train_labels, k):
    hyps = []
    for test_id in tqdm(range(len(test_embeddings))):
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
                dists.append([dist[j], train_labels[i*batch_size + j]])

        sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)[:k]
        all_indices = [obj[1] for obj in sorted_dists]
        c = Counter(all_indices)
        majority = c.most_common()[0][0]
        hyps.append(majority)
    return hyps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--src_lang", type=str, default="x", help="source language")
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
    print("cross:", args.cross)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("prompt:", args.prompt)
    print("###########################")

    set_seed(args.seed)

    if args.cross:
        output_dir = f"outputs/save_classification_cross_{args.src_lang}"
    else:
        output_dir = "outputs/save_classification"

    if "embed-multilingual" in args.model_checkpoint:
        model = cohere.Client(COHERE_TOKEN)
        batch_size = 64
    elif "text-embedding-3-large" in args.model_checkpoint:
        model = OpenAI(api_key=OPENAI_TOKEN)
        batch_size = 64
    else:
        model = SentenceTransformer(args.model_checkpoint).cuda()
        batch_size = 128

    if args.dataset == "nusax":
        dataset = NusaXDataset(prompt=args.prompt, task="classification")
    if args.dataset == "lince_sa":
        dataset = LinceSADataset(prompt=args.prompt)
    if args.dataset == "massive_intent":
        dataset = MassiveIntentDataset(prompt=args.prompt)
    if args.dataset == "sib200":
        dataset = Sib200Dataset(prompt=args.prompt)
    if args.dataset == "nollysenti":
        dataset = NollySentiDataset(prompt=args.prompt, task="classification")
    if args.dataset == "mtop_intent":
        dataset = MTOPIntentDataset(prompt=args.prompt)
    if args.dataset == "fire":
        dataset = FIREDataset(prompt=args.prompt)

    for lang in dataset.LANGS:
        if args.cross and lang == args.src_lang:
            print("skip src language eval", lang)
            continue
            
        # get embeddings
        if args.cross:
            train_texts = dataset.train_data[args.src_lang]["source"]
            train_labels = dataset.train_data[args.src_lang]["target"]
        else:
            train_texts = dataset.train_data[lang]["source"]
            train_labels = dataset.train_data[lang]["target"]

        if len(train_texts) < batch_size:
            batch_size = len(train_texts) // 2
        num_of_batches = len(train_texts) // batch_size

        if (len(train_texts) % batch_size) > 0:
            num_of_batches += 1

        if args.cross:
            print("> train:", args.src_lang, num_of_batches)
        else:
            print("> train:", lang, num_of_batches)
        train_embeddings = []
        for i in tqdm(range(num_of_batches)):
            train_batch_text = train_texts[i*batch_size:(i+1)*batch_size]
            train_batch_label = train_labels[i*batch_size:(i+1)*batch_size]

            if "embed-multilingual" in args.model_checkpoint:
                train_batch_embeddings = get_cohere_embedding(model, train_batch_text, args.model_checkpoint)
            elif "text-embedding-3-large" in args.model_checkpoint:
                train_batch_embeddings = get_openai_embedding(model, train_batch_text, args.model_checkpoint)
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

            if "embed-multilingual" in args.model_checkpoint:
                test_batch_embeddings = get_cohere_embedding(model, test_batch_text, args.model_checkpoint)
            elif "text-embedding-3-large" in args.model_checkpoint:
                test_batch_embeddings = get_openai_embedding(model, test_batch_text, args.model_checkpoint)
            else:
                test_batch_embeddings = model.encode(test_batch_text, normalize_embeddings=False)

            if len(test_embeddings) == 0:
                test_embeddings = test_batch_embeddings
            else:
                for emb in test_batch_embeddings:
                    test_embeddings = np.concatenate((test_embeddings, np.expand_dims(emb, axis=0)), axis=0)

        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.model_checkpoint}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.model_checkpoint}/seed_{args.seed}/")

        for k in [1,5,10]:
            key = lang
            print(key, k, train_embeddings.shape, test_embeddings.shape)
            hyps = evaluate_classification(train_embeddings, test_embeddings, train_labels, k=k)
            # print(hyps)
            # print(test_labels)
            obj = {}
            obj[f'acc'] = accuracy_score(test_labels, hyps)
            obj[f'prec'] = precision_score(test_labels, hyps, average="macro")
            obj[f'rec'] = recall_score(test_labels, hyps, average="macro")
            obj[f'f1'] = f1_score(test_labels, hyps, average="macro")
            print(obj)

            file_path = output_dir + "/" + args.dataset + "/" + args.model_checkpoint + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)