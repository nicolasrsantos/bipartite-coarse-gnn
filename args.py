from argparse import ArgumentParser

def args_train():
    parser = ArgumentParser()

    # model and dataset args
    parser.add_argument("--x_dim", dest="x_dim", type=int)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int)
    parser.add_argument("--out_dim", dest="out_dim", type=int)
    parser.add_argument(
        "--coarsened", dest="coarsened", action="store_true",
        help="whether the graph used is coarsened or not"
    )
    parser.add_argument(
        "--coarse_level", dest="coarse_level", type=int,
        help="level of the coarsened graph used. must be specified if coarsened=True.",
        default=None
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        help="use R8, R52, ohsumed or 20ng"
    )
    parser.add_argument(
        "--model", dest="model", type=str,
        help="GNN model. GCN, GraphSAGE and GAT supported"
    )

    # training args
    parser.add_argument("--batch_size", dest="batch_size", type=int)
    parser.add_argument("--gpu", dest="cuda", action="store_true",
                        help="whether to train using gpu")
    parser.add_argument("--cpu", dest="cuda", action="store_false",
                        help="whether to train using cpu")
    parser.add_argument("--n_epochs", dest="n_epochs", type=int)
    parser.add_argument("--lr", dest="lr", type=float)
    parser.add_argument("--epoch_log", dest="epoch_log", type=int)
    parser.add_argument("--n_runs", dest="n_runs", type=int,
                        help="number of train runs with the same config")
    parser.add_argument(
        "--patience", dest="patience", type=int,
        help="number of epochs without validation loss improvement "
        " before early stopping"
    )

    # dir args
    parser.add_argument(
        "--graphs_dir", dest="graphs_dir", type=str,
        help="directory to save the edge_index, edge_weight, x, y, and masks"
    )
    parser.add_argument(
        "--models_dir", dest="models_dir", type=str,
        help="dir to store saved models"
    )
    parser.add_argument(
        "--train_test_info_dir", dest="train_test_info_dir", type=str,
        help="directory to get class information about the docs"
    )

    parser.set_defaults(
        model="sage", batch_size=32, x_dim=300, hidden_dim=256, cuda=True,
        coarsened=False, n_epochs=200, lr=1e-3, epoch_log=10, patience=10,
        n_runs=10, graphs_dir="data/graphs/", models_dir="models/",
        train_test_info_dir="data/train_test_info/"
    )

    args = parser.parse_args()
    return args


def args_build_graph():
    parser = ArgumentParser()

    # dataset args
    parser.add_argument("--x_dim", dest="x_dim", type=int)
    parser.add_argument("--val_split", dest="val_split", type=float)
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        help="use R8, R52, ohsumed or 20ng"
    )
    parser.add_argument(
        "--emb_file", dest="emb_file", type=str,
        help="usage example 'glove.6B.300d' for glove's 300d 6B file. please "
        "make sure to set feature_dim with the same dimension of the embedding"
    )

    # dir args
    parser.add_argument("--cleaned_dir", dest="cleaned_dir", type=str,
                        help="directory 8of the cleaned data files")
    parser.add_argument("--info_dir", dest="info_dir", type=str,
                        help="directory to get class information about the docs")
    parser.add_argument("--corpus_dir", dest="corpus_dir", type=str,
                        help="directory where the raw corpora files are stored")
    parser.add_argument("--emb_dir", dest="emb_dir", type=str,
                        help="directory where the embedding files are stored")
    parser.add_argument(
        "--graphs_dir", dest="graphs_dir", type=str,
        help="directory to save the edge_index, edge_weight, x, y, and masks"
    )


    parser.set_defaults(
        dataset="R8", x_dim=300, val_split=0.1, emb_file="glove.6B.300d",
        x_init="word", cleaned_dir="data/cleaned/", emb_dir="embeddings/",
        info_dir="data/train_test_info/", graphs_dir="data/graphs/",
        corpus_dir="data/corpus/"
    )

    args = parser.parse_args()
    return args

def args_coarse():
    parser = ArgumentParser()

    # model and dataset args
    parser.add_argument("--x_dim", dest="x_dim", type=int)
    parser.add_argument(
        "--max_level", dest="max_level", type=int,
        help="maximum level achieved during coarsening. arg used to process all "
        "levels from 1 to max_level. must be executed in order to run the GNN on each level"
    )
    parser.add_argument(
        "--dataset", dest="dataset", type=str,
        help="use R8, R52, ohsumed or 20ng"
    )

    # dir args
    parser.add_argument(
        "--graphs_dir", dest="graphs_dir", type=str,
        help="directory to save the edge_index, edge_weight, x, y, and masks"
    )
    parser.add_argument(
        "--mfbn_dir", dest="mfbn_dir", type=str,
        help="dir where coarsening tool (mfbn) is stored"
    )

    parser.set_defaults(
        x_dim=300, mfbn_dir="/home/nicolas/code/mfbn/input/",
        graphs_dir="data/graphs/", models_dir="models/",
        train_test_info_dir="data/train_test_info/",
        max_level=None
    )

    args = parser.parse_args()
    return args
