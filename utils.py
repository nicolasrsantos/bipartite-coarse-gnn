import numpy as np
import random
import pickle as pkl
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


class RunInfo():
    def __init__(self, run_id, args):
        self.run_id = run_id
        self.dataset = args.dataset
        self.lr = args.lr
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.n_runs = args.n_runs
        self.args = args
        self.acc_mean = None
        self.f1_mean = None
        self.acc_std = None
        self.f1_std = None
        self.runtime = None
        self.avg_mem_allocated = None
        self.avg_mem_reserved = None
        self.experiments = []


    def save_run_info(self):
        run_dir = self.args.models_dir + self.dataset + "/"
        run_args = str(self.lr) + "_" + str(self.hidden_dim) + "_" + str(self.batch_size)
        filename = run_args + "_" + str(self.run_id) + ".run_info"

        with open(run_dir + filename, "wb") as f:
            pkl.dump(self, f)


class ExperimentInfo():
    def __init__(self, experiment_id, saved_model, args):
        self.experiment_id = experiment_id
        self.saved_model = saved_model
        self.last_epoch = None
        self.best_model_epoch = None
        self.test_metrics = {}
        self.history = {
            "tr_loss":[],
            "tr_acc":[],
            "tr_f1":[],
            "val_loss":[],
            "val_acc":[],
            "val_f1":[]
        }
        self.training_time = []
        self.memory_usage = {
            "max_mem_allocated": None,
            "max_mem_reserved": None,
        }


def read_file(file_dir: str, filename: str) -> list:
    file_content = []
    with open(file_dir + filename + ".txt", "r") as f:
        for line in f.readlines():
            file_content.append(line)
    print(f"read a file with {len(file_content)} elements")

    return file_content


def read_embedding_file(emb_dir, emb_file):
    embeddings = {}
    with open(emb_dir + emb_file + ".txt", "r") as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            emb_vec = [float(i) for i in data[1:]]
            embeddings[word] = emb_vec
    print(f"read an embedding file with {len(list(embeddings.keys()))} elements")
    return embeddings


def create_rand_features(data, low=-0.01, high=0.01, dim=300):
    random_features = {}

    if isinstance(data, dict):
        data = list(data.keys())
    elif isinstance(data, set):
        data = list(data)

    for i in range(len(data)):
        random_features[data[i]] = np.random.uniform(low, high, dim)

    return random_features


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_graph(graph_dict, args):
    Path(args.graphs_dir).mkdir(parents=True, exist_ok=True)
    save_str = args.graphs_dir + args.dataset + "_" + str(args.x_dim)

    with open(save_str + ".edge_index", "wb") as f:
        pkl.dump(graph_dict["edge_index"], f)
    with open(save_str + ".x_doc", "wb") as f:
        pkl.dump(graph_dict["x_doc"], f)
    with open(save_str + ".x_word", "wb") as f:
        pkl.dump(graph_dict["x_word"], f)
    with open(save_str + ".y", "wb") as f:
        pkl.dump(graph_dict["y"], f)
    with open(save_str + ".masks", "wb") as f:
        pkl.dump(graph_dict["masks"], f)
    with open(save_str + ".embs", "wb") as f:
        pkl.dump(graph_dict["embs"], f)
    with open(save_str + ".word_id_map", "wb") as f:
        pkl.dump(graph_dict["word_id_map"], f)

    print("graph information saved")


def nx_to_edge_idx(G):
    source = []
    target = []

    for edge in G.edges(data=True):
        if isinstance(edge[0], str):
            i, j = edge[0].split("_"), edge[1]
        else:
            i, j = edge[1].split("_"), edge[0]
        i = int(i[1])
        source.append(i)
        target.append(j)
        source.append(j)
        target.append(i)
    edge_index = [source, target]

    return edge_index


def nx_to_node_list(projected_nodes):
    nodes = set()
    for node in projected_nodes:
        node = node.split("_")
        node = int(node[1])
        nodes.add(node)

    return nodes

def read_graph(args):
    read_str = args.graphs_dir + args.dataset + "_" + str(args.x_dim)

    graph_dict = {}
    if args.coarsened:
        with open(read_str + "-" + str(args.coarse_level) + ".coarsened_edge_index", "rb") as f:
            graph_dict["edge_index"] = pkl.load(f)
        with open(read_str + "-" + str(args.coarse_level) + ".coarsened_x_word", "rb") as f:
            graph_dict["x_word"] = pkl.load(f)
    else:
        with open(read_str + ".edge_index", "rb") as f:
            graph_dict["edge_index"] = pkl.load(f)
        with open(read_str + ".x_word", "rb") as f:
            graph_dict["x_word"] = pkl.load(f)
    with open(read_str + ".x_doc", "rb") as f:
        graph_dict["x_doc"] = pkl.load(f)
    with open(read_str + ".y", "rb") as f:
        graph_dict["y"] = pkl.load(f)
    with open(read_str + ".masks", "rb") as f:
        graph_dict["masks"] = pkl.load(f)

    return graph_dict

def prepare_graph(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)

    x_doc = np.array(graph_dict["x_doc"])
    x_doc = torch.tensor(x_doc, dtype=torch.float)
    x_word = np.array(graph_dict["x_word"])
    x_word = torch.tensor(x_word, dtype=torch.float)
    y = torch.tensor(graph_dict["y"], dtype=torch.long)

    n_doc_nodes = len(y)
    train_idx, val_idx, test_idx = graph_dict["masks"][0], graph_dict["masks"][1], graph_dict["masks"][2]
    train_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_doc_nodes, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx] = 1
    test_mask[test_idx] = 1

    data = HeteroData()
    data["doc"].x = x_doc
    data["doc"].y = y
    data["doc"].train_mask = train_mask
    data["doc"].val_mask = val_mask
    data["doc"].test_mask = test_mask
    data["word"].x = x_word
    data["doc", "has_word", "word"].edge_index = edge_index
    data["word", "is_in_doc", "doc"].edge_index = edge_index.flip(0)

    return data

def check_graph_properties(G):
    print("checking graph properties")
    print(f"\tGraph has isolated nodes: {G.has_isolated_nodes()}")
    print(f"\tGraph has self loops: {G.has_self_loops()}")
    print(f"\tGraph is undirected: {G.is_undirected()}")