import pickle as pkl
import json

from utils import *
from args import *

def to_interval(val, old_interval, new_interval):
    new_val = (((val-old_interval[0])*(new_interval[1]-new_interval[0]))/(old_interval[1]-old_interval[0]))+new_interval[0]
    return int(new_val)

set_seed(42)

def run(args):
    dataset = args.dataset
    read_str = "data/graphs/" + dataset + "_300"

    for coarse_level in range(1, args.max_level + 1):
        with open(read_str + ".embs", "rb") as f:
            embs = pkl.load(f)
        with open(read_str + ".word_id_map", "rb") as f:
            word_id_map = pkl.load(f)
        with open(read_str + "-" + str(coarse_level) + ".ncol", "r") as f:
            adj_list = f.readlines()
        with open(read_str + "-" + str(coarse_level) + ".membership", "r") as f:
            membership = f.readlines()
        with open(read_str + "-" + str(coarse_level) + ".weight", "r") as f:
            weight = f.readlines()
        with open(read_str + "-" + str(coarse_level) + "-info.json") as f:
            coarse_info = json.load(f)

        sv_map = {}
        membership = [int(el) for el in membership]
        for i, node in enumerate(membership):
            if node in sv_map:
                sv_map[node].append(i)
            else:
                sv_map[node] = [i]

        old_int = [min(list(word_id_map.values())), max(list(word_id_map.values()))]
        new_int = [coarse_info["source_vertices"][0], len(membership)-1]

        id_word_map = {}
        for k, v in word_id_map.items():
            new_key = to_interval(v, old_int, new_int)
            id_word_map[new_key] = k

        x_sv = []
        word_start = coarse_info["coarsened_vertices"][0]
        for sv in sv_map.keys():
            if sv >= word_start:
                sv_emb = []
                for v in sv_map[sv]:
                    word = id_word_map[v]
                    sv_emb.append(embs[word])
                sv_emb = np.mean(sv_emb, axis=0)
                x_sv.append(np.array(sv_emb))
        x_sv = np.array(x_sv)


        src, tgt, coarsened_edge_weight = [], [], []
        for edge in adj_list:
            edge = edge.split()
            i, j, w = int(edge[0]), int(edge[1]), int(edge[2])
            src.append(i)
            tgt.append(j)
            coarsened_edge_weight.append(w)

        before = [min(tgt), max(tgt)]
        after = [0, coarse_info["coarsened_vertices"][1] - 1]
        print(before, after)

        tgt = [to_interval(v, before, after) for v in tgt]
        coarsened_edge_index = [src, tgt]

        uniques = {}
        for edge in zip(coarsened_edge_index[0], coarsened_edge_index[1]):
            if edge in uniques:
                uniques[edge] += 1
            else:
                uniques[edge] = 1
        print(len(uniques), coarse_info["coarsened_ecount"], len(coarsened_edge_index[0]), len(coarsened_edge_index[1]), len(x_sv))
        assert len(uniques) == coarse_info["coarsened_ecount"]

        save_str = "data/graphs/" + dataset
        with open(save_str + "_300-" + str(coarse_level) + ".coarsened_edge_index", "wb") as f:
            pkl.dump(coarsened_edge_index, f)
        with open(save_str + "_300-" + str(coarse_level) + ".coarsened_x_word", "wb") as f:
            pkl.dump(x_sv, f)
        with open(save_str + "_300-" + str(coarse_level) + ".coarsened_edge_weight", "wb") as f:
            pkl.dump(coarsened_edge_weight, f)

if __name__ == "__main__":
    args = args_coarse()
    print(args)
    if args.max_level == None:
        raise TypeError("please set max_level arg in order to process the coarsened graph")

    run(args)