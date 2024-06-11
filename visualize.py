import os
import torch
import random
import numpy as np
import argparse
import cohere
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import NusaXDataset, LinceSADataset, MassiveIntentDataset, Sib200Dataset, NollySentiDataset, MTOPIntentDataset, FIREDataset

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
    parser.add_argument("--dataset", type=str, default="mtop", help="snips or mtop or multi-nlu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prompt", type=str, default="", help="prompt")
    parser.add_argument("--num_color", type=int, default=200, help="number of colors")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    print("dataset:", args.dataset)
    print("model_checkpoint:", args.model_checkpoint)
    print("seed:", args.seed)
    print("cuda:", args.cuda)
    print("verbose:", args.verbose)
    print("fp16:", args.fp16)
    print("prompt:", args.prompt)
    print("num_color:", args.num_color)
    print("###########################")

    set_seed(args.seed)

    output_dir = "outputs/visualization/"

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

    num_colors = args.num_color
    if args.dataset == "sib200":
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(20000)]
    else:
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(200)]

    class_colors = ["#0041c2","#06d6a0","#ff5400","#457b9d","#efdf48","#8bd346","#9b5fe0"]

    dataset_size = []
    train_embeddings = []
    train_all_labels = []
    train_langs = []
    for lang in dataset.LANGS:
        train_texts = dataset.train_data[lang]["source"][:num_colors]
        train_labels = dataset.train_data[lang]["target"][:num_colors]

        for _ in train_texts:
            train_langs.append(lang)

        train_all_labels = train_all_labels + train_labels

        dataset_size.append(len(train_texts))

        if len(train_texts) < batch_size:
            batch_size = len(train_texts) // 2
        num_of_batches = len(train_texts) // batch_size

        if (len(train_texts) % batch_size) > 0:
            num_of_batches += 1

        print("> train:", lang, num_of_batches)
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

    mpl.rc('xtick', labelsize=18) 
    mpl.rc('ytick', labelsize=18) 
    
    # TSNE
    X_embedded = TSNE(n_components=2, init="pca", random_state=args.seed).fit_transform(train_embeddings)
    print(X_embedded.shape)
    print("dataset_size:", dataset_size)

    X_embedded = (X_embedded-np.min(X_embedded))/(np.max(X_embedded)-np.min(X_embedded)) 
    
    start_id, end_id = 0, 0
    for i in range(len(dataset_size)):
        if i == 0:
            start_id = 0
            end_id = dataset_size[0]
        else:
            start_id += dataset_size[i]
            end_id += dataset_size[i]
        plt.scatter(X_embedded[start_id:end_id,0], X_embedded[start_id:end_id,1], s=15, color=colors[i]);    

    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    model_short_name = args.model_checkpoint.split("/")[-1]
    output_path = f"{output_dir}/tsne_{args.dataset}_{model_short_name}_language_color.png"

    plt.savefig(output_path, dpi=200)

    plt.clf() 
    
    for i in range(num_colors):
        arr_x = []
        arr_y = []
        for j in range(len(dataset_size)):
            arr_x.append(X_embedded[j*num_colors+i,0])
            arr_y.append(X_embedded[j*num_colors+i,1])
        
        plt.scatter(arr_x, arr_y, s=15, color=colors[i]);    
    plt.show()
    
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    model_short_name = args.model_checkpoint.split("/")[-1]
    output_path = f"{output_dir}/tsne_{args.dataset}_{model_short_name}.png"
    
    plt.savefig(output_path, dpi=300)

    plt.clf() 
    
    for i in range(len(X_embedded)):        
        # print(">", class_colors[train_all_labels[i]])
        plt.scatter([X_embedded[i,0]], [X_embedded[i,1]], s=15, color=class_colors[train_all_labels[i]]);    
    plt.show()
    
    if not os.path.exists(f"{output_dir}"):
        os.makedirs(f"{output_dir}")

    model_short_name = args.model_checkpoint.split("/")[-1]
    output_path = f"{output_dir}/tsne_{args.dataset}_{model_short_name}_class.png"
    
    plt.savefig(output_path, dpi=100)

    if args.dataset == "sib200":
        plt.clf() 
        
        for i in range(len(X_embedded)):   
            lang_family = dataset.LANGS_TO_LANGS_FAMILY_ID[train_langs[i]]
            plt.scatter([X_embedded[i,0]], [X_embedded[i,1]], s=15, color=colors[lang_family]);    
        plt.show()
        
        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")
    
        model_short_name = args.model_checkpoint.split("/")[-1]
        output_path = f"{output_dir}/tsne_{args.dataset}_{model_short_name}_language_family.png"
        # plt.legend(bbox_to_anchor=(1.01, 1.03))
        # plt.tight_layout()
        plt.savefig(output_path, dpi=300)
