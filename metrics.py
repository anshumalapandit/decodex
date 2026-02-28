import pandas as pd
import networkx as nx


# ---------------------------------------------------
# 1. COUNTERPARTY CONCENTRATION
# ---------------------------------------------------

def compute_counterparty_concentration(df):
    """
    For each client, compute:
    Volume with top counterparty / total volume
    """

    # Total trades per client
    total_trades = (
        df.groupby("buyer").size()
        .add(df.groupby("seller").size(), fill_value=0)
    )

    # Pair frequency
    pair_counts = (
        df.groupby(["buyer", "seller"])
        .size()
        .reset_index(name="trade_count")
    )

    results = []

    for client in total_trades.index:

        buyer_pairs = pair_counts[pair_counts["buyer"] == client][["seller", "trade_count"]]

        seller_pairs = pair_counts[pair_counts["seller"] == client][["buyer", "trade_count"]]
        seller_pairs.columns = ["seller", "trade_count"]

        all_pairs = pd.concat([buyer_pairs, seller_pairs])

        if len(all_pairs) == 0:
            continue

        top_trade = all_pairs["trade_count"].max()
        total = total_trades[client]

        concentration = top_trade / total

        results.append({
            "client": client,
            "counterparty_concentration": concentration,
            "total_trades": total
        })

    return pd.DataFrame(results)


# ---------------------------------------------------
# 2. RECIPROCITY SCORE
# ---------------------------------------------------

def compute_reciprocity(df):
    """
    Measure proportion of reciprocal trade pairs.
    """

    pair_counts = (
        df.groupby(["buyer", "seller"])
        .size()
        .reset_index(name="trade_count")
    )

    pair_set = set(zip(pair_counts["buyer"], pair_counts["seller"]))

    reciprocal_count = 0

    for buyer, seller in pair_set:
        if (seller, buyer) in pair_set:
            reciprocal_count += 1

    return reciprocal_count / len(pair_set) if len(pair_set) > 0 else 0


# ---------------------------------------------------
# 3. BUILD NETWORK GRAPH
# ---------------------------------------------------

def build_trade_graph(df):
    """
    Create directed graph from trades.
    """

    G = nx.DiGraph()

    for _, row in df.iterrows():
        buyer = row["buyer"]
        seller = row["seller"]

        if G.has_edge(buyer, seller):
            G[buyer][seller]["weight"] += 1
        else:
            G.add_edge(buyer, seller, weight=1)

    return G


# ---------------------------------------------------
# 4. TRIANGLE LOOP DETECTION
# ---------------------------------------------------

def detect_triangular_loops(G):
    """
    Detect 3-node cycles (A→B→C→A)
    """

    cycles = list(nx.simple_cycles(G))

    triangles = [cycle for cycle in cycles if len(cycle) == 3]

    return triangles