# Standard imports
import pandas as pd
import numpy as np
import networkx as nx
import random
from collections import defaultdict
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# Imports requiring packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel


# -----------------------
# DATA LOADER
# -----------------------
def load_data(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    print(df.columns)
    return df

# -----------------------
# GRAPH BUILDER
# -----------------------
def build_transaction_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['from'], row['to'], amount=row['amount'], timestamp=row['timestamp'])
    return G

# -----------------------
# LABEL GENERATOR
# -----------------------
def assign_node_labels(df):
    from_labels = df[['from', 'fromIsPhi']].groupby('from').max().to_dict()['fromIsPhi']
    to_labels = df[['to', 'toIsPhi']].groupby('to').max().to_dict()['toIsPhi']
    labels = defaultdict(int)
    for node in set(from_labels) | set(to_labels):
        labels[node] = max(from_labels.get(node, 0), to_labels.get(node, 0))
    return labels

# -----------------------
# FEATURE ENGINEERING
# -----------------------
def compute_all_node_features(G):
    features = {}
    for node in G.nodes():
        try:
            degree = G.degree(node)
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            in_sum = sum(G[u][node].get("amount", 0) for u in G.predecessors(node))
            out_sum = sum(G[node][v].get("amount", 0) for v in G.successors(node))
            total_sum = in_sum + out_sum
            neighbors = set(G.predecessors(node)) | set(G.successors(node))
            tx_count = len(neighbors)
            zero_tx = sum(1 for n in neighbors if G.get_edge_data(node, n, {}).get("amount", 1) == 0)
            zero_tx_ratio = zero_tx / (tx_count + 1e-6)
            inv_freq = 1.0 / (tx_count + 1e-6)
            features[node] = np.array([
                degree, in_deg, out_deg, total_sum, out_sum, in_sum,
                len(neighbors), inv_freq, zero_tx_ratio, tx_count
            ], dtype=np.float32)
        except Exception:
            continue
    return features

# -----------------------
# GAT ENCODER
# -----------------------
class GATEncoder(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=32, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, 1)

    def forward(self, g, x):
        x = self.gat1(g, x)
        x = x.flatten(1)
        x = self.gat2(g, x)
        return x.mean(1)

# -----------------------
# CONTRASTIVE LOSS
# -----------------------
def info_nce_loss(z1, z2, temperature=0.4):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_matrix = torch.mm(z1, z2.T)
    positives = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    negatives = torch.sum(torch.exp(sim_matrix / temperature), dim=1)
    loss = -torch.log(positives / negatives)
    return loss.mean()

# -----------------------
# GRAPH UTILITIES
# -----------------------
def get_ego_graph(G, node, radius=2):
    return nx.ego_graph(G, node, radius=radius, center=True, undirected=False)

def random_walk_with_restart(G, start_node, restart_prob=0.8, walk_length=15):
    walk = [start_node]
    current = start_node
    for _ in range(walk_length - 1):
        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        if not neighbors or random.random() < restart_prob:
            current = start_node
        else:
            current = random.choice(neighbors)
        walk.append(current)
    return list(set(walk))

def sample_subgraphs(G, node):
    ego = get_ego_graph(G, node)
    sub1 = G.subgraph(random_walk_with_restart(ego, node)).copy()
    sub2 = G.subgraph(random_walk_with_restart(ego, node)).copy()
    return sub1, sub2

# -----------------------
# TRAINING
# -----------------------
def contrastive_train(G, labels_dict, encoder, features_dict, device, epochs=5, sample_size=300):
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    nodes = list(labels_dict.keys())
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(nodes)
        for node in tqdm(nodes[:min(sample_size, len(nodes))], desc=f"Epoch {epoch+1}"):
            if node not in features_dict:
                continue
            try:
                g1, g2 = sample_subgraphs(G, node)
                f1 = torch.tensor([features_dict[n] for n in g1.nodes() if n in features_dict], dtype=torch.float32).to(device)
                f2 = torch.tensor([features_dict[n] for n in g2.nodes() if n in features_dict], dtype=torch.float32).to(device)
                if f1.size(0) == 0 or f2.size(0) == 0:
                    continue
                g1 = dgl.add_self_loop(dgl.from_networkx(g1)).to(device)
                g2 = dgl.add_self_loop(dgl.from_networkx(g2)).to(device)
                z1 = encoder(g1, f1)
                z2 = encoder(g2, f2)
                if z1.shape[0] != z2.shape[0]:
                    continue
                loss = info_nce_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Training error at node {node}: {e}")
                continue
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss / sample_size:.4f}")

# -----------------------
# EMBEDDING EXTRACTION
# -----------------------
def extract_embeddings(G, nodes, encoder, features_dict, device):
    encoder.eval()
    embs = []
    valid_nodes = []
    for node in tqdm(nodes, desc="Extracting Embeddings"):
        if node not in features_dict:
            continue
        try:
            ego = get_ego_graph(G, node)
            valid = [n for n in ego.nodes() if n in features_dict]
            if node not in valid:
                continue
            feats = torch.tensor([features_dict[n] for n in valid], dtype=torch.float32).to(device)
            g_dgl = dgl.add_self_loop(dgl.from_networkx(G.subgraph(valid))).to(device)
            z = encoder(g_dgl, feats)
            idx = valid.index(node)
            embs.append(z[idx].detach().cpu().numpy())
            valid_nodes.append(node)
        except Exception as e:
            print(f"Embedding error at node {node}: {e}")
    return np.array(embs), valid_nodes


# -----------------------
# CLASSIFICATION
# -----------------------
def train_classifier(embeddings, labels_dict, nodes):
    console = Console()
    y = [labels_dict[n] for n in nodes]
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = XGBClassifier(n_estimators=100)
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    total_nodes = len(y_test)
    total_normal = sum(1 for label in y_test if label == 0)
    total_phishing = sum(1 for label in y_test if label == 1)

    console.print(Panel.fit(
        f"[bold]Combined Inductive Test Results for ALL test months:[/bold]\n"
        f"[cyan]Total nodes in test data:[/] {total_nodes}\n"
        f"[cyan]Total NORMAL nodes (true):[/] {total_normal}\n"
        f"[cyan]Total PHISHING nodes (true):[/] {total_phishing}",
        title="Test Summary", border_style="bright_blue"
    ))

    table = Table(title="Confusion Matrix", box=box.SQUARE, border_style="yellow")
    table.add_column("", justify="center", style="bold")
    table.add_column("Predicted Normal", justify="center")
    table.add_column("Predicted Phishing", justify="center")
    table.add_row("Actual Normal", f"{tn}", f"{fp}")
    table.add_row("Actual Phishing", f"{fn}", f"{tp}")
    console.print(table)

    report_dict = classification_report(
        y_test, y_pred, target_names=['Normal', 'Phishing'], output_dict=True, digits=4
    )
    report_table = Table(title="📊 Classification Report", box=box.MINIMAL_DOUBLE_HEAD, border_style="green")
    report_table.add_column("Class", justify="center", style="bold")
    report_table.add_column("Precision", justify="center")
    report_table.add_column("Recall", justify="center")
    report_table.add_column("F1-Score", justify="center")
    report_table.add_column("Support", justify="center")

    for cls in ["Normal", "Phishing"]:
        metrics = report_dict[cls]
        report_table.add_row(
            cls,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1-score']:.4f}",
            f"{int(metrics['support'])}"
        )

    for avg in ["accuracy", "macro avg", "weighted avg"]:
        metrics = report_dict.get(avg, {})
        if avg == "accuracy":
            report_table.add_row(
                avg.title(),
                "-",
                "-",
                f"{metrics:.4f}" if isinstance(metrics, float) else "-",
                f"{total_nodes}"
            )
        else:
            report_table.add_row(
                avg.title(),
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                "-"
            )
    console.print(report_table)

    normal_correct = tn
    phishing_correct = tp
    normal_percent = (normal_correct / total_normal * 100) if total_normal else 0
    phishing_percent = (phishing_correct / total_phishing * 100) if total_phishing else 0

    console.print(
        f"[green]✅ Normal correctly classified:[/] [bold]{normal_correct}[/] / {total_normal} = [bold]{normal_percent:.2f}%[/]"
    )
    console.print(
        f"[green]✅ Phishing correctly classified:[/] [bold]{phishing_correct}[/] / {total_phishing} = [bold]{phishing_percent:.2f}%[/]"
    )


# -----------------------
# MAIN
# -----------------------
def main():
    path = "final_data_6.csv"
    df = load_data(path)
    G = build_transaction_graph(df)
    labels = assign_node_labels(df)
    features_dict = compute_all_node_features(G)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = GATEncoder().to(device)
    contrastive_train(G, labels, encoder, features_dict, device)
    embeddings, final_nodes = extract_embeddings(G, list(labels.keys()), encoder, features_dict, device)
    train_classifier(embeddings, labels, final_nodes)

if __name__ == "__main__":
    main()
