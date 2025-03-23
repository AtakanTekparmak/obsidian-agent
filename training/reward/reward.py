import os
import re
import torch

from transformers import AutoTokenizer, AutoModel

from training.obsidian.utils import build_graph_from_vault

# Import all needed metrics from metrics.py
from training.reward.metrics import (
    directed_modularity,
    directed_clustering_coefficient,
    directed_betweenness_centrality,
    largest_weakly_connected_fraction,
    degree_distribution_heterogeneity,
    note_cohesion,
    link_relevance,
    folder_cohesion,
    uniqueness
)

def compute_stage1_reward(vault_path: str) -> float:
    """
    Computes the Stage 1 reward, which is purely structural:
      1) Directed modularity
      2) Directed clustering coefficient
      3) Directed betweenness centrality (avg)
      4) Largest weakly connected component fraction
      5) Degree distribution heterogeneity (in + out)

    Returns a float in [0,1] which is the average of these five metrics.
    """
    G = build_graph_from_vault(vault_path)
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    
    # 1. Directed modularity
    mod_val = directed_modularity(G)
    # 2. Directed clustering
    cluster_val = directed_clustering_coefficient(G)
    # 3. Directed betweenness
    bc_val = directed_betweenness_centrality(G)
    # 4. Weak connectivity fraction
    conn_val = largest_weakly_connected_fraction(G)
    # 5. Degree distribution heterogeneity
    hetero_val = degree_distribution_heterogeneity(G)
    
    metrics = [mod_val, cluster_val, bc_val, conn_val, hetero_val]
    return sum(metrics) / len(metrics)

# Prepare a global ModernBERT model for embeddings (avoid reloading each call)
MODERNBERT_MODEL_NAME = "nomic-ai/modernbert-embed-base"
_tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL_NAME)
_model = AutoModel.from_pretrained(MODERNBERT_MODEL_NAME)
_model.eval()

def embed_text(text: str):
    """
    Embed a text string using ModernBERT. Returns a 1D torch.Tensor.
    Uses mean pooling over valid tokens.
    """
    inputs = _tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    # outputs.last_hidden_state: [batch_size=1, seq_len, hidden_dim]
    token_embeddings = outputs.last_hidden_state[0]  # shape [seq_len, hidden_dim]
    attention_mask = inputs['attention_mask'][0]     # shape [seq_len]
    valid_tokens = token_embeddings[attention_mask.bool()]
    if valid_tokens.shape[0] == 0:
        return torch.zeros(outputs.last_hidden_state.shape[-1])
    return valid_tokens.mean(dim=0)


def compute_stage2_reward(vault_path: str) -> float:
    """
    Computes the Stage 2 reward by combining:
    - The same 5 structural metrics as Stage 1
    - Plus 4 semantic metrics:
        6) Note cohesion
        7) Link relevance
        8) Folder cohesion
        9) Uniqueness (lack of near-duplicate notes)

    Returns a float in [0,1] which is the average of the 9 metrics.
    """
    # Build directed graph
    G = build_graph_from_vault(vault_path)
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    
    #===============================
    # 1-5: Structural metrics
    #===============================
    mod_val = directed_modularity(G)
    cluster_val = directed_clustering_coefficient(G)
    bc_val = directed_betweenness_centrality(G)
    conn_val = largest_weakly_connected_fraction(G)
    hetero_val = degree_distribution_heterogeneity(G)
    
    #===============================
    # Semantic Preprocessing
    #===============================
    # For each note, read its entire text, embed it fully (for link_relevance/folder_cohesion)
    # and also embed by sentence (for note_cohesion).
    
    note_texts = {}
    vault_abs = os.path.abspath(vault_path)
    for node_id in G.nodes():
        full_path = os.path.join(vault_abs, node_id)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                note_texts[node_id] = f.read().strip()
        except:
            note_texts[node_id] = ""
    
    # 1) Full-note embeddings
    note_embeddings = {}
    for nid, text in note_texts.items():
        if not text:
            note_embeddings[nid] = torch.zeros(_model.config.hidden_size)
        else:
            note_embeddings[nid] = embed_text(text)
    
    # 2) Sentence-level embeddings for note cohesion
    note_sentence_embeddings = {}
    for nid, text in note_texts.items():
        if not text:
            note_sentence_embeddings[nid] = []
            continue
        # simple sentence splitting:
        sentences = re.split(r'(?<=[.\n])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_embs = [embed_text(s) for s in sentences]
        note_sentence_embeddings[nid] = sent_embs
    
    # Also build folder_dict for folder cohesion
    folder_dict = {}
    for node_id in G.nodes():
        folder = os.path.dirname(node_id)  # the parent directory
        if folder not in folder_dict:
            folder_dict[folder] = []
        folder_dict[folder].append(node_id)
    
    #===============================
    # 6) Note Cohesion
    #===============================
    note_cohesion_val = note_cohesion(note_sentence_embeddings)
    
    #===============================
    # 7) Link Relevance
    #===============================
    link_relevance_val = link_relevance(G, note_embeddings)
    
    #===============================
    # 8) Folder Cohesion
    #===============================
    folder_cohesion_val = folder_cohesion(folder_dict, note_embeddings)
    
    #===============================
    # 9) Uniqueness
    #===============================
    uniqueness_val = uniqueness(note_embeddings, threshold=0.95)
    
    # Combine all 9
    metrics = [
        mod_val,
        cluster_val,
        bc_val,
        conn_val,
        hetero_val,
        note_cohesion_val,
        link_relevance_val,
        folder_cohesion_val,
        uniqueness_val
    ]
    
    # Clamp to [0,1] just in case
    final_vals = []
    for x in metrics:
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0
        final_vals.append(x)
    
    # Average
    return sum(final_vals) / len(final_vals)