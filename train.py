import time
import datetime
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero

from utils import *
from args import *


def get_model(hidden_dim, out_dim):
    model = Sequential("x, edge_index", [
        (SAGEConv((-1, -1), hidden_dim), "x, edge_index -> x"),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), hidden_dim), "x, edge_index -> x"),
        ReLU(inplace=True),
        (Linear(-1, out_dim), "x -> x"),
    ])
    return model


@torch.no_grad()
def init_params(train_loader, model, device):
    batch = next(iter(train_loader))
    batch = batch.to(device)
    model(batch.x_dict, batch.edge_index_dict)


def train(train_loader, model, optimizer, device):
    model.train()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        batch = batch.to(device)
        batch_size = batch["doc"].batch_size

        out = model(batch.x_dict, batch.edge_index_dict)["doc"][:batch_size]
        loss = F.cross_entropy(out, batch["doc"].y[:batch_size])
        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch["doc"].y[:batch_size].cpu().numpy())

        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        train_loss = total_loss / total_examples
    train_acc = accuracy_score(y_true, y_pred)
    train_f1 = f1_score(y_true, y_pred, average="macro")

    return train_loss, train_acc, train_f1


@torch.no_grad()
def eval(loader, model, device):
    model.eval()

    y_pred, y_true = [], []
    total_examples = total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch["doc"].batch_size

        out = model(batch.x_dict, batch.edge_index_dict)["doc"][:batch_size]
        loss = F.cross_entropy(out, batch["doc"].y[:batch_size])
        preds = F.softmax(out, dim=-1)
        preds = preds.argmax(dim=-1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(batch["doc"].y[:batch_size].cpu().numpy())

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        eval_loss = total_loss / total_examples
    eval_acc = accuracy_score(y_true, y_pred)
    eval_f1 = f1_score(y_true, y_pred, average="macro")

    return eval_loss, eval_acc, eval_f1


def experiment(train_loader, val_loader, test_loader, model, args):
    torch.cuda.reset_peak_memory_stats()
    device = "cpu"
    if args.cuda and torch.cuda.is_available():
        device = "cuda"

    model = model.to(device)
    init_params(train_loader, model, device) # warm-up for benchmarking
    torch.cuda.synchronize() # wait for warm-up to complete entirely

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    experiment_id = str(time.time())
    saved_model = args.models_dir + args.dataset + "/" + experiment_id + ".pt"
    exp_info = ExperimentInfo(experiment_id, saved_model, args)
    es_count, best_vloss = 0, float('inf') # early stopping

    print("starting training")
    times = []
    for epoch in range(1, args.n_epochs + 1):
        torch.cuda.synchronize()

        if es_count == args.patience:
            exp_info.last_epoch = epoch - 1
            print(f"early stopping on epoch {epoch}")
            break

        start_epoch = time.time()
        tr_loss, tr_acc, tr_f1 = train(train_loader, model, optimizer, device)
        torch.cuda.synchronize()
        times.append(time.time() - start_epoch)

        val_loss, val_acc, val_f1 = eval(val_loader, model, device)
        exp_info.history["tr_loss"].append(tr_loss), exp_info.history["val_loss"].append(val_loss)
        exp_info.history["tr_acc"].append(tr_acc), exp_info.history["val_acc"].append(val_acc)
        exp_info.history["tr_f1"].append(tr_f1), exp_info.history["val_f1"].append(val_f1)

        if epoch == 1 or epoch % args.epoch_log == 0:
            print(
                f"epoch {epoch} | loss {tr_loss:.4f} | acc {tr_acc:.4f} | f1 {tr_f1:.4f}"
                f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}"
            )

        if val_loss <= best_vloss:
            es_count = 0
            best_vloss = val_loss
            exp_info.best_model_epoch = epoch - 1 # 0-based idx to use history array
            torch.save(model.state_dict(), saved_model)
        else:
            es_count += 1
        exp_info.last_epoch = epoch

    model.load_state_dict(torch.load(saved_model))
    test_loss, test_acc, test_f1 = eval(test_loader, model, device)
    print(f"test_loss {test_loss:.4f} | test_acc {test_acc:.4f} | test_f1 {test_f1:.4f}")
    exp_info.test_metrics = {"loss":test_loss, "acc":test_acc, "f1":test_f1}

    exp_info.training_time = times
    exp_info.memory_usage["max_mem_allocated"] = torch.cuda.max_memory_allocated()
    exp_info.memory_usage["max_mem_reserved"] = torch.cuda.max_memory_reserved()

    return exp_info


def run(args):
    set_seed(42)

    if args.dataset is None:
        raise TypeError("dataset arg was not set.")
    if args.out_dim is None:
        raise TypeError("output dimension was not set.")
    if args.coarsened is None:
        raise TypeError("coarsened arg was not set.")
    if args.coarsened and args.coarse_level is None:
        raise TypeError("input is a coarsened graph but coarse level was not set.")

    graph_dict = read_graph(args)
    graph = prepare_graph(graph_dict)
    check_graph_properties(graph)

    train_nodes = ("doc", graph["doc"].train_mask)
    val_nodes = ("doc", graph["doc"].val_mask)
    test_nodes = ("doc", graph["doc"].test_mask)

    train_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=train_nodes,
        batch_size=args.batch_size
    )
    val_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=val_nodes,
        batch_size=args.batch_size
    )
    test_loader = NeighborLoader(
        graph,
        num_neighbors=[-1] * 2,
        input_nodes=test_nodes,
        batch_size=args.batch_size
    )

    print(
        f"train loader size: {len(train_loader)}\n"
        f"val loader size: {len(val_loader)}\n"
        f"test loader size: {len(test_loader)}"
    )

    model = get_model(args.hidden_dim, args.out_dim)
    model = to_hetero(model, graph.metadata(), aggr="sum")

    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir + args.dataset).mkdir(parents=True, exist_ok=True)
    run_id = str(time.time())
    run_info = RunInfo(run_id, args)
    acc, f1 = [], []
    max_mem_alloc, max_mem_reserved, runtime = [], [], []

    print("stating experiment")
    for i in range(args.n_runs):
        print(f"run number: {i + 1}")

        exp_info = experiment(train_loader, val_loader, test_loader, model, args)

        run_info.experiments.append(exp_info)
        acc.append(exp_info.test_metrics["acc"])
        f1.append(exp_info.test_metrics["f1"])

        max_mem_alloc.append(exp_info.memory_usage["max_mem_allocated"])
        max_mem_reserved.append(exp_info.memory_usage["max_mem_reserved"])
        runtime.append(np.sum(exp_info.training_time))

    acc_mean, acc_std = np.mean(acc), np.std(acc)
    f1_mean, f1_std = np.mean(f1), np.std(f1)
    run_info.acc_mean, run_info.acc_std = acc_mean, acc_std
    run_info.f1_mean, run_info.f1_std = f1_mean, f1_std

    run_info.runtime = runtime
    run_info.avg_mem_alloc = np.mean(max_mem_alloc)/1024**2
    run_info.avg_mem_reserved = np.mean(max_mem_reserved)/1024**2
    run_info.save_run_info()

    elapsed = str(datetime.timedelta(seconds=np.sum(runtime)))
    print(
        f"final acc: {acc_mean:.4f} | acc_std: {acc_std:.4f}\n"
        f"final f1: {f1_mean:.4f} | f1_std: {f1_std:.4f}\n"
        f"runtime: {elapsed}\n"
        f"avg memory allocated: {run_info.avg_mem_alloc:.2f} MB\n"
        f"avg memory reserved: {run_info.avg_mem_reserved:.2f} MB"
    )

if __name__ == "__main__":
    args = args_train()
    print(args)
    run(args)