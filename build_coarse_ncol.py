import pickle as pkl

from args import *

def to_interval(val, old_interval, new_interval):
    new_val = (((val-old_interval[0])*(new_interval[1]-new_interval[0]))/(old_interval[1]-old_interval[0]))+new_interval[0]
    return int(new_val)


def run(args):
    if args.dataset is None:
        raise Exception("dataset was not specified")

    print("reading edge index")
    read_str = args.graphs_dir + args.dataset + "_" + str(args.x_dim)
    with open(read_str + ".edge_index", "rb") as f:
        edge_index = pkl.load(f)

    print("converting data")
    doc, words = list(set(edge_index[0])), list(set(edge_index[1]))
    old_interval = [0, max(words)]
    new_interval = [len(doc), len(doc)+max(words)] # or len(doc)+len(words)-1
    print(old_interval, new_interval)
    word_int_map = {
        word:to_interval(word, old_interval, new_interval) for word in list(dict.fromkeys(words))
    }

    edge_list = []
    for i, j in zip(edge_index[0], edge_index[1]):
        edge_list.append([i, word_int_map[j], 1.0])

    print("saving ncol")
    with open(args.mfbn_dir + args.dataset + ".ncol", "w") as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")

if __name__ == "__main__":
    args = args_coarse()
    print(args)
    run(args)