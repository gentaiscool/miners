import os
import torch
import random
import numpy as np
import argparse
import json
import cohere
import transformers
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from collections import Counter

from utils import NusaXDataset, NusaTranslationDataset, TatoebaDataset, BUCCDataset, LinceMTDataset, PhincDataset, LinceSADataset, MassiveIntentDataset, Sib200Dataset, NollySentiDataset, MTOPIntentDataset, FIREDataset

from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import hashlib

OPENAI_TOKEN = ""
COHERE_TOKEN = ""
HF_TOKEN = ""

def argmax(array):
    """argmax with deterministic pseudorandom tie breaking."""
    max_indices = np.arange(len(array))[array == np.max(array)]
    idx = int(hashlib.sha256(np.asarray(array).tobytes()).hexdigest(),16) % len(max_indices)
    return max_indices[idx]

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def normalize(x):
    x = np.array(x)
    return np.exp(x - logsumexp(x))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_commandr_chat_response(gen_model, gen_model_checkpoint, text, seed):
    response = gen_model.chat(
        model="command-r",
        message=text,
        temperature=0,
        max_tokens=64,
        seed=seed,
        p=1
    )
    return response.text


def get_mt0_response(gen_model, tokenizer, gen_model_checkpoint, text, seed):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(gen_model.device)

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )

    response = outputs[0]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_gemma_response(gen_model, tokenizer, gen_model_checkpoint, text, seed):
    messages = [
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(gen_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_mistral_instruct_chat_response(gen_model, tokenizer, gen_model_checkpoint, text, seed):
    messages = [
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(gen_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_llama3_instruct_chat_response(gen_model, tokenizer, gen_model_checkpoint, text, seed):
    messages = [
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(gen_model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=1
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_openai_chat_response(gen_model, gen_model_checkpoint, text, seed):
    messages=[
        {
            "role": "user",
            "content": text
        }
    ]
    response = gen_model.chat.completions.create(
        model=gen_model_checkpoint,
        messages=messages,
        temperature=0,
        max_tokens=64,
        top_p=1,
        seed=seed
    )
    return response.choices[0].message.content

def get_openai_embedding(model, texts, checkpoint="text-embedding-3-large"):
    data = model.embeddings.create(input = texts, model=checkpoint).data
    embeddings = []
    for obj in data:
        embeddings.append(obj.embedding)
    return embeddings

def get_cohere_embedding(model, texts, model_checkpoint):
    response = model.embed(texts=texts, model=model_checkpoint, input_type="search_query")
    return response.embeddings

def retrieve_ids(train_embeddings, test_embeddings, train_labels, k, balance=False, all_possible_labels=[]):
    all_samples = []
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
                dists.append([dist[j], train_labels[i*batch_size + j], i*batch_size + j])

        if balance:
            sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)
        else:
            sorted_dists = sorted(dists,key=lambda l:l[0], reverse=False)[:k]

        all_indices = []
        if balance:
            # print(all_possible_labels)
            for opt in all_possible_labels:
                count_found = 0
                # print(">", opt)
                for obj in sorted_dists:
                    # print(">>", opt, obj[1])
                    if opt == obj[1]:
                        all_indices.append(obj[2])
                        count_found += 1
                        if count_found == k:
                            break
        else:
            all_indices = [obj[2] for obj in sorted_dists]
        # print(">", len(all_indices))
        all_samples.append(all_indices)
    return all_samples


def calculate_log_prob(model, tokenizer, prefix, targets):
    log_sums = []
    for target in targets:
        # Encode input and output
        # print(">", prefix)
        # print("target>", target)
        input_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
        output_tokens = tokenizer.encode(target, add_special_tokens=False, return_tensors='pt')

        # Concatenate input and output tokens
        tokens = torch.cat([input_tokens, output_tokens], dim=1)

        # Get model predictions for the entire sequence at once
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits

        log_sum = 0
        range_index = range(input_tokens.shape[1] - 1, tokens.shape[1] - 1)
        len_range = tokens.shape[1] - 1 - (input_tokens.shape[1] - 1) 
        for i in range_index:
            past_tok, current_tok = i, i + 1
            token_logit = logits[0, past_tok, :]
            token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
            log_token_prob = token_log_probs[tokens[0, current_tok]].item()
            log_sum += log_token_prob

            token = tokenizer.decode(tokens[:, current_tok])
            # print(f"Token: {token}, Log Prob: {log_token_prob}")

        # print(">", output_tokens, log_sum, input_tokens.shape[1] - 1, tokens.shape[1] - 1, len_range)
        log_sums.append(log_sum / len_range)

    normalized_scores = normalize(log_sums)
    pred = targets[argmax(normalized_scores)]
    return pred, normalized_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model")
    parser.add_argument(
        "--gen_model_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--src_lang", type=str, default="eng", help="source language")
    parser.add_argument("--dataset", type=str, default="mtop", help="snips or mtop or multi-nlu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--load_in_8bit", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    parser.add_argument("--instruction", type=str, default="", help="instruction")
    parser.add_argument("--k", type=int, default=1, help="k")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    print("src_lang:", args.src_lang)
    print("dataset:", args.dataset)
    print("model_checkpoint:", args.model_checkpoint)
    print("gen_model_checkpoint:", args.gen_model_checkpoint)
    print("seed:", args.seed)
    print("cuda:", args.cuda)
    print("cross:", args.cross)
    print("verbose:", args.verbose)
    print("prompt:", args.prompt)
    print("balance:", args.balance)
    print("instruction:", args.instruction)
    print("load_in_8bit:", args.load_in_8bit)
    print("k:", args.k)
    print("###########################")

    set_seed(args.seed)

    if args.load_in_8bit:
        output_dir = "logs/save_icl_8bit"
    else:
        output_dir = "logs/save_icl"

    if args.cross:
        output_dir = output_dir + f"_cross_{args.src_lang}"

    if args.balance:
        output_dir = output_dir + "_balance"

    if "embed-multilingual" in args.model_checkpoint:
        model = cohere.Client(COHERE_TOKEN)
        batch_size = 64
    elif "text-embedding-3-large" in args.model_checkpoint:
        model = OpenAI(api_key=OPENAI_TOKEN)
        batch_size = 64
    else:
        model = SentenceTransformer(args.model_checkpoint).cuda()
        batch_size = 128

    if "mistralai/Mistral-7B-Instruct-v0.3" in args.gen_model_checkpoint or "Meta-Llama-3-" in args.gen_model_checkpoint or "google/gemma-1.1-7b-it" in args.gen_model_checkpoint or "Meta-Llama-3.1-" in args.gen_model_checkpoint:
        if args.load_in_8bit:
            gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
    elif "CohereForAI/aya-101" in args.gen_model_checkpoint or "bigscience/mt0" in args.gen_model_checkpoint:
        if args.load_in_8bit:
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
        else:
            gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
    elif "facebook/xglm" in args.gen_model_checkpoint or "bigscience/bloomz" in args.gen_model_checkpoint or "aya-23-8B" in args.gen_model_checkpoint:
        if args.load_in_8bit:
            gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN, device_map="auto", load_in_8bit=True)
        else:
            gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(args.gen_model_checkpoint, token=HF_TOKEN)
    elif "gpt-3.5-turbo" in args.gen_model_checkpoint or "gpt-4" in args.gen_model_checkpoint:
        gen_model = OpenAI(api_key=OPENAI_TOKEN)
    elif "command-r" in args.gen_model_checkpoint:
        gen_model = cohere.Client(COHERE_TOKEN)
    elif "Meta-Llama-3.1" in args.gen_model_checkpoint:
        gen_model = transformers.pipeline(
            "text-generation",
            model=args.gen_model_checkpoint,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=HF_TOKEN
        )

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
            train_labels = dataset.train_data[args.src_lang]["target_text"]
        else:
            train_texts = dataset.train_data[lang]["source"]
            train_labels = dataset.train_data[lang]["target_text"]

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
        test_labels = dataset.test_data[lang]["target_text"]

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

        if not os.path.exists(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/"):
            os.makedirs(f"{output_dir}/{args.dataset}/{args.gen_model_checkpoint}/{args.model_checkpoint}/seed_{args.seed}/")

        for k in [args.k]:
            key = lang
            print(key, k, train_embeddings.shape, test_embeddings.shape)
            if k > 0:
                if args.balance:
                    print("> balance")
                    all_few_shot_samples_ids = retrieve_ids(train_embeddings, test_embeddings, train_labels, k=k, all_possible_labels=dataset.TEXT_LABELS, balance=True)
                else:
                    all_few_shot_samples_ids = retrieve_ids(train_embeddings, test_embeddings, train_labels, k=k)

            hyps = []
            prompts = []
            for text_id in tqdm(range(len(test_texts))):
                text = "Instruction:" + args.instruction + "\n"

                if k > 0:
                    few_shot_text = ""
                    # print(all_few_shot_samples_ids)
                    for few_shot_sample_id in all_few_shot_samples_ids[text_id]:
                        few_shot_text += "Input:" + train_texts[few_shot_sample_id] + " Prediction:" + train_labels[few_shot_sample_id] + "\n"
                    text += few_shot_text + "\n"
                if "CohereForAI/aya-101" in args.gen_model_checkpoint or "bigscience/mt0" in args.gen_model_checkpoint or "gpt-3.5-turbo" in args.gen_model_checkpoint or "gpt-4" in args.gen_model_checkpoint or "command-r" in args.gen_model_checkpoint or "Meta-Llama-3-" in args.gen_model_checkpoint or "mistralai/Mistral-7B-Instruct-v0.3" in args.gen_model_checkpoint or "google/gemma-1.1-7b-it" in args.gen_model_checkpoint or "Meta-Llama-3.1" in args.gen_model_checkpoint:
                    text += "Options:" + str(dataset.TEXT_LABELS) + "\n"
                text += "Input:" + test_texts[text_id] + " Prediction:"

                targets = dataset.TEXT_LABELS
                if "gpt-3.5-turbo" in args.gen_model_checkpoint or "gpt-4" in args.gen_model_checkpoint:
                    hyp = get_openai_chat_response(gen_model, args.gen_model_checkpoint, text, args.seed)
                elif "command-r" in args.gen_model_checkpoint:
                    hyp = get_commandr_chat_response(gen_model, args.gen_model_checkpoint, text, args.seed)
                elif "mistralai/Mistral-7B-Instruct-v0.3" in args.gen_model_checkpoint:
                    hyp = get_mistral_instruct_chat_response(gen_model, tokenizer, args.gen_model_checkpoint, text, args.seed)
                elif "google/gemma-1.1-7b-it" in args.gen_model_checkpoint:
                    hyp = get_gemma_response(gen_model, tokenizer, args.gen_model_checkpoint, text, args.seed)
                elif "Meta-Llama-3-" in args.gen_model_checkpoint:
                    hyp = get_llama3_instruct_chat_response(gen_model, tokenizer, args.gen_model_checkpoint, text, args.seed)
                elif "Meta-Llama-3.1-" in args.gen_model_checkpoint:
                    hyp = get_llama3_instruct_chat_response(gen_model, tokenizer, args.gen_model_checkpoint, text, args.seed)
                elif "CohereForAI/aya-101" in args.gen_model_checkpoint or "bigscience/mt0" in args.gen_model_checkpoint:
                    hyp = get_mt0_response(gen_model, tokenizer, args.gen_model_checkpoint, text, args.seed)
                else:
                    hyp, normalized_scores = calculate_log_prob(gen_model, tokenizer, text, dataset.TEXT_LABELS)
                hyps.append(hyp)
                prompts.append(text)

            if "google/gemma-1.1-7b-it" in args.gen_model_checkpoint or "gpt-3.5-turbo" in args.gen_model_checkpoint or "command-r" in args.gen_model_checkpoint:
                lower_test_labels = [x.lower() for x in test_labels]
                lower_hyps = [x.lower() for x in hyps]

                obj = {}
                obj[f'acc'] = accuracy_score(lower_test_labels, lower_hyps)
                obj[f'prec'] = precision_score(lower_test_labels, lower_hyps, average="macro")
                obj[f'rec'] = recall_score(lower_test_labels, lower_hyps, average="macro")
                obj[f'f1'] = f1_score(lower_test_labels, lower_hyps, average="macro")
                print(obj)

                preds = {"hyp":hyps, "gold":test_labels, "lower_hyp": lower_hyps, "lower_gold": lower_test_labels}
            else:
                obj = {}
                obj[f'acc'] = accuracy_score(test_labels, hyps)
                obj[f'prec'] = precision_score(test_labels, hyps, average="macro")
                obj[f'rec'] = recall_score(test_labels, hyps, average="macro")
                obj[f'f1'] = f1_score(test_labels, hyps, average="macro")
                print(obj)

                preds = {"hyp":hyps, "gold":test_labels}
            all_prompts = {"prompts":prompts}

            file_path = output_dir + "/" + args.dataset + "/" + args.gen_model_checkpoint + "/" + args.model_checkpoint + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + ".json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(obj, outfile, indent=4)

            file_path = output_dir + "/" + args.dataset + "/" + args.gen_model_checkpoint + "/" + args.model_checkpoint + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + "_preds.json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(preds, outfile, indent=4)

            file_path = output_dir + "/" + args.dataset + "/" + args.gen_model_checkpoint + "/" + args.model_checkpoint + "/" + "/seed_" + str(args.seed) + "/eval_" + key + "_" + str(k) + "_prompts.json"
            print("writing results to file_path:", file_path)
            with open(file_path, "w") as outfile: 
                json.dump(all_prompts, outfile, indent=4)
