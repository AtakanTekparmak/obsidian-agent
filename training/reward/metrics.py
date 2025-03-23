import math
import networkx as nx
import torch
import re

def directed_modularity(G):
    """
    Computes a directed modularity score for the graph G using a community-detection
    algorithm (e.g., Louvain). Modularity measures how well the graph is partitioned
    into clusters with dense intra-links and sparse inter-links.
    
    Returns a float in [0,1], clamping any negative modularity to 0.
    """
    if G.number_of_edges() == 0:
        return 0.0
    communities = nx.community.louvain_communities(G, weight=None, seed=42)
    mod_value = nx.community.modularity(G, communities, resolution=1)
    if mod_value < 0:
        mod_value = 0.0
    if mod_value > 1.0:
        mod_value = 1.0
    return mod_value

def directed_clustering_coefficient(G):
    """
    Computes the (average) directed clustering coefficient for the graph G.
    The clustering coefficient indicates the prevalence of closed triads in a directed
    network, measuring how interconnected each node's neighbors are.
    
    Returns a float in [0,1], where 1 means maximum local clustering (highly interlinked)
    and 0 means no triangular closures.
    """
    if G.number_of_nodes() == 0:
        return 0.0
    c = nx.average_clustering(G)
    if c < 0:
        c = 0.0
    if c > 1.0:
        c = 1.0
    return c

def directed_betweenness_centrality(G):
    """
    Computes the average normalized betweenness centrality across all nodes in G, 
    treating edges as directed when searching for shortest paths.
    
    Betweenness centrality measures the extent to which nodes bridge different areas
    of the graph. A higher average betweenness suggests that, on average, nodes 
    participate more in shortest paths, encouraging 'hub' or 'bridge' notes.

    Returns a float in [0,1].
    """
    if G.number_of_nodes() == 0:
        return 0.0
    bc = nx.betweenness_centrality(G, normalized=True)
    if len(bc) == 0:
        return 0.0
    avg_bc = sum(bc.values()) / len(bc)
    if avg_bc < 0:
        avg_bc = 0.0
    if avg_bc > 1.0:
        avg_bc = 1.0
    return avg_bc

def largest_weakly_connected_fraction(G):
    """
    Computes the fraction of nodes in the largest weakly connected component 
    of the directed graph G. This indicates how well the graph is connected
    if we ignore edge direction (weak connectivity).

    Returns a float in [0,1], 1 meaning all nodes are reachable (as an undirected set)
    and <1 means the vault is split into isolated components.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    fraction = len(largest_wcc) / n
    return fraction

def gini_coefficient(values):
    """
    Computes the Gini coefficient (0..1) for a list of non-negative values.
    A value of 0 means perfect equality, 1 means maximum inequality.
    Used to measure how uneven the distribution of in/out-degree is.
    """
    if len(values) == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    N = len(values)
    cum_sum = 0.0
    for i, val in enumerate(sorted_vals, start=1):
        cum_sum += i * val
    gini = (2.0 * cum_sum) / (N * total) - (N + 1.0) / N
    if gini < 0:
        gini = 0.0
    if gini > 1.0:
        gini = 1.0
    return gini

def degree_distribution_heterogeneity(G):
    """
    Measures how uneven (heterogeneous) the in- and out-degrees in the directed graph G are.
    We compute the Gini coefficient for both in-degree and out-degree, then average them.
    
    Returns a float in [0,1]. A higher value means a more skewed degree distribution
    (some notes have many links, some have few), which can encourage hub-like structures.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    out_degs = [d for _, d in G.out_degree()]
    in_degs = [d for _, d in G.in_degree()]
    g_out = gini_coefficient(out_degs)
    g_in = gini_coefficient(in_degs)
    return (g_out + g_in) / 2.0

def note_cohesion(note_sentence_embeddings):
    """
    Computes the average intra-note cohesion across all notes.
    
    note_sentence_embeddings: Dict[str, List[torch.Tensor]]
        A mapping from note_id -> list of sentence embeddings (torch.Tensor).
    
    For each note with >=2 sentences, we compute pairwise cosine similarities among 
    the embeddings. We then average those per-note similarities, and finally average 
    across all notes to get the final note cohesion score in [0,1].
    """
    all_cohesions = []
    for node_id, embeddings in note_sentence_embeddings.items():
        if len(embeddings) < 2:
            # Treat single-sentence notes as fully cohesive
            continue
        # Normalize each sentence embedding
        vecs = torch.stack(embeddings)
        vecs = torch.nn.functional.normalize(vecs, dim=1)
        sim_matrix = torch.mm(vecs, vecs.T)
        n_sents = len(embeddings)
        # Sum upper triangular (excluding diagonal) to get average pairwise similarity
        total_sim = sim_matrix.triu(diagonal=1).sum().item()
        num_pairs = n_sents * (n_sents - 1) / 2
        all_cohesions.append(total_sim / num_pairs)
    if len(all_cohesions) == 0:
        return 1.0  # If all notes have <2 sentences, define cohesion as 1
    avg_cohesion = sum(all_cohesions) / len(all_cohesions)
    # clamp to [0,1] just in case
    if avg_cohesion < 0:
        avg_cohesion = 0.0
    if avg_cohesion > 1:
        avg_cohesion = 1.0
    return avg_cohesion

def link_relevance(G, note_embeddings):
    """
    Computes the average cosine similarity between source and target notes
    for every directed edge in G. 

    note_embeddings: Dict[str, torch.Tensor]
        A mapping from note_id -> single note embedding (averaged over the entire note).

    Returns a float in [0,1], where a higher value means links connect more related content.
    If there are no edges, returns 0.
    """
    edges = list(G.edges())
    if len(edges) == 0:
        return 0.0
    sims = []
    for u, v in edges:
        u_emb = torch.nn.functional.normalize(note_embeddings[u], dim=0)
        v_emb = torch.nn.functional.normalize(note_embeddings[v], dim=0)
        cos_sim = torch.dot(u_emb, v_emb).item()
        sims.append(cos_sim)
    avg_sim = sum(sims) / len(sims)
    if avg_sim < 0:
        avg_sim = 0.0
    if avg_sim > 1.0:
        avg_sim = 1.0
    return avg_sim

def folder_cohesion(folder_dict, note_embeddings):
    """
    Computes how cohesive each folder is, by averaging pairwise cosine similarities
    between notes in the same folder, then returns the overall average across folders.
    
    folder_dict: Dict[str, List[str]]
        Maps folder_path -> list of note_ids in that folder
    note_embeddings: Dict[str, torch.Tensor]
        Mapping note_id -> embedding

    Returns a float in [0,1]. Folders with <2 notes are skipped or treated as cohesive.
    """
    folder_sims = []
    for folder, note_ids in folder_dict.items():
        if len(note_ids) < 2:
            continue
        local_pairs = 0
        local_sum = 0.0
        for i in range(len(note_ids)):
            for j in range(i+1, len(note_ids)):
                emb_i = torch.nn.functional.normalize(note_embeddings[note_ids[i]], dim=0)
                emb_j = torch.nn.functional.normalize(note_embeddings[note_ids[j]], dim=0)
                sim = torch.dot(emb_i, emb_j).item()
                local_sum += sim
                local_pairs += 1
        if local_pairs > 0:
            folder_sims.append(local_sum / local_pairs)
    if len(folder_sims) == 0:
        return 1.0  # If all folders have <2 notes, define folder cohesion as 1
    fc = sum(folder_sims) / len(folder_sims)
    if fc < 0:
        fc = 0.0
    if fc > 1.0:
        fc = 1.0
    return fc

def uniqueness(note_embeddings, threshold=0.95):
    """
    Identifies near-duplicate notes (cosine similarity > threshold) and computes a
    uniqueness score = 1 - (#near_duplicates / total_pairs).

    note_embeddings: Dict[str, torch.Tensor]
        A mapping from note_id -> single note embedding
    threshold: float
        The similarity cutoff above which two notes are considered near-duplicates

    Returns a float in [0,1]. A higher value means fewer duplicates.
    """
    nodes_list = list(note_embeddings.keys())
    n = len(nodes_list)
    if n < 2:
        return 1.0  # trivially unique if there's only 0 or 1 note
    # normalize embeddings for each note
    norm_emb = {}
    for node_id in nodes_list:
        norm_emb[node_id] = torch.nn.functional.normalize(note_embeddings[node_id], dim=0)
    near_dup_count = 0
    total_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i+1, n):
            sim = torch.dot(norm_emb[nodes_list[i]], norm_emb[nodes_list[j]]).item()
            if sim > threshold:
                near_dup_count += 1
    score = 1.0 - (near_dup_count / total_pairs)
    if score < 0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score