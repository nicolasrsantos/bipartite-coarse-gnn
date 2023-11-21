import random
import networkx as nx

from utils import *
from args import *

VALID_DATASETS = ["20ng", "ohsumed", "R8", "R52", "mr", "ag_news", "SST1", "SST2", "TREC", "WebKB"]

def get_data(args):
    if args.dataset not in VALID_DATASETS:
        raise Exception(
            "dataset not valid.\n supported datasets {VALID_DATASETS}"
        )
    cleaned_dataset = args.dataset + ".clean"
    embs = None

    dataset = read_file(args.cleaned_dir, cleaned_dataset)
    tr_tst_info = read_file(args.info_dir, args.dataset)
    if args.x_init == "word":
        embs = read_embedding_file(args.emb_dir, args.emb_file)

    return dataset, tr_tst_info, embs


def get_word_nodes(dataset):
    word_nodes = set()
    for doc in dataset:
        doc_words = doc.split()
        word_nodes.update(doc_words)
    word_nodes = list(word_nodes)

    return word_nodes


def build_graph(dataset, y_map, doc_name, embs, args):
    graph_dict = {}
    word_nodes = get_word_nodes(dataset)
    word_id_map = {word:i for i, word in enumerate(word_nodes)}
    graph_dict["word_id_map"] = word_id_map

    G = nx.Graph()
    for doc_id, doc in enumerate(dataset):
        doc_words = doc.split()
        doc_id = "doc_" + str(doc_id)
        G.add_node(doc_id, bipartite=0)
        for word in doc_words:
            word_id = word_id_map[word]
            G.add_node(word_id, bipartite=1)
            G.add_edge(doc_id, word_id)

    print(f"dataset has {len(dataset)} docs and {len(get_word_nodes(dataset))} words")

    docs, words = [], []
    for edge in G.edges(data=True):
        if isinstance(edge[0], str):
            i, j = edge[0].split("_"), edge[1]
        else:
            i, j = edge[1].split("_"), edge[0]
        i = int(i[1])
        docs.append(i)
        words.append(j)
    graph_dict["edge_index"] = [docs, words]

    print("building feature vectors")
    x_doc = []
    for _ in range(len(dataset)):
        x_doc.append(np.random.uniform(-.01, .01, args.x_dim))
    graph_dict["x_doc"] = x_doc

    oov = create_rand_features(word_nodes)
    for key, value in oov.items():
        if key not in embs:
            embs[key] = value
    graph_dict["embs"] = embs

    x_word = []
    for i, word in enumerate(word_nodes):
        x_word.append(embs[word])
    graph_dict["x_word"] = x_word

    y = []
    for i in range(len(dataset)):
        doc_meta = doc_name[i].split('\t')
        label = doc_meta[2]
        y.append(y_map[label])
    graph_dict["y"] = y

    return graph_dict


def run(args):
    set_seed(42)

    dataset, tr_ts_info, embs = get_data(args)

    doc_name_list = []
    doc_train_list, doc_test_list = [], []
    for tti in tr_ts_info:
        doc_name_list.append(tti.strip())
        temp = tti.split()

        if temp[1].find("train") != -1:
            doc_train_list.append(tti.strip())
        if temp[1].find("test") != -1:
            doc_test_list.append(tti.strip())

    train_ids = []
    for train_name in doc_train_list:
        train_id = doc_name_list.index(train_name)
        train_ids.append(train_id)
    random.shuffle(train_ids)

    test_ids = []
    for test_name in doc_test_list:
        test_id = doc_name_list.index(test_name)
        test_ids.append(test_id)
    random.shuffle(test_ids)

    ids = train_ids + test_ids

    # shuffle dataset
    shuffle_doc_name_list = []
    shuffle_dataset = []
    for id in ids:
        shuffle_doc_name_list.append(doc_name_list[int(id)])
        shuffle_dataset.append(dataset[int(id)])

    # Get labels
    y = []
    for doc_meta in shuffle_doc_name_list:
        temp = doc_meta.split("\t")
        y.append(temp[2])
    y_map = {label:i for i, label in enumerate(set(y))}

    train_size = len(train_ids)
    val_size = int(args.val_split * train_size)
    real_train_size = train_size - val_size

    masks_train = ids[0:real_train_size]
    masks_val = ids[real_train_size:real_train_size+val_size]
    masks_test = ids[train_size:]
    print(len(masks_train), len(masks_val), len(masks_test))

    graph_dict = build_graph(
        shuffle_dataset, y_map, shuffle_doc_name_list, embs, args
    )
    graph_dict["masks"] = [masks_train, masks_val, masks_test]

    print(f"saving {args.dataset}'s graph information to : {args.graphs_dir}")
    save_graph(graph_dict, args)


if __name__ == "__main__":
    args = args_build_graph()
    print(args)
    run(args)