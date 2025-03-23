import networkx as nx
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def visualize_obsidian_graph(G, output_path=None, figsize=(12, 10), 
                            node_size=200, node_color_by='folder',
                            layout_type='spring'):
    """
    Visualizes a NetworkX graph of an Obsidian vault.
    
    Parameters:
    -----------
    G : nx.DiGraph
        The graph to visualize, as returned by build_graph_from_vault()
    output_path : str or None
        If provided, saves the visualization to this path. Otherwise displays it.
    figsize : tuple
        Figure size as (width, height) in inches
    node_size : int or list
        Size of nodes. Can be a single value or list of sizes for each node.
    node_color_by : str
        How to color nodes: 'folder' (by folder) or 'connections' (by number of connections)
    layout_type : str
        Layout algorithm to use: 'spring', 'kamada_kawai', 'circular', 'random'
        
    Returns:
    --------
    fig, ax
        The matplotlib figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout algorithm
    if layout_type == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'random':
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)  # default
    
    # Define node color schemes
    if node_color_by == 'folder':
        # Group nodes by folder
        folder_groups = defaultdict(list)
        for node in G.nodes():
            folder = os.path.dirname(node) or "root"
            folder_groups[folder].append(node)
        
        # Create color map for folders
        import matplotlib.colors as mcolors
        color_list = list(mcolors.TABLEAU_COLORS.values())
        folder_colors = {}
        for i, folder in enumerate(folder_groups.keys()):
            folder_colors[folder] = color_list[i % len(color_list)]
        
        # Assign colors to nodes based on folder
        node_colors = []
        for node in G.nodes():
            folder = os.path.dirname(node) or "root"
            node_colors.append(folder_colors[folder])
    else:  # color by number of connections
        # Color by degree (total connections)
        degrees = dict(G.degree())
        node_colors = [degrees[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='gray', arrows=True, ax=ax)
    
    # Add node labels (titles instead of paths)
    labels = {node: G.nodes[node].get('title', os.path.basename(node).replace('.md', '')) 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)
    
    # Add title and remove axes
    plt.title("Obsidian Vault Graph Visualization", fontsize=16)
    plt.axis('off')
    
    # Add legend for folder colors if applicable
    if node_color_by == 'folder':
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, 
                                 label=folder.split('/')[-1] if folder != 'root' else 'root')
                          for folder, color in folder_colors.items()]
        ax.legend(handles=legend_elements, title="Folders", loc='upper right')
    
    # Save or show the figure
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def build_graph_from_vault(vault_path: str) -> nx.DiGraph:
    """
    Parses an Obsidian vault directory (vault_path) to create a directed graph of notes.
    Each .md file is treated as a node, and any [[TargetNote]] references in the text
    create a directed edge from the source note to 'TargetNote'.
    
    Returns a networkx.DiGraph with node attributes:
      - 'title': the note title (filename without .md)
    Edges are directed from the source note to the note referenced in [[...]].
    """
    G = nx.DiGraph()
    note_paths = {}  # Map note title -> node_id (relative path)
    
    vault_path = os.path.abspath(vault_path)
    # Scan all .md files
    for root, _, files in os.walk(vault_path):
        for fname in files:
            if fname.endswith(".md"):
                note_title = fname[:-3]  # remove '.md'
                rel_path = os.path.relpath(os.path.join(root, fname), vault_path)
                node_id = rel_path.replace("\\", "/")  # normalize path
                G.add_node(node_id, title=note_title)
                # if the note title is unique, store a path reference
                if note_title not in note_paths:
                    note_paths[note_title] = node_id
    
    # Regex for [[Note Title]] or [[Note Title|Alias]]
    link_pattern = re.compile(r"\[\[([^|\]]+)(?:\|[^]]*)?\]\]")
    
    # Read each file, find links, create directed edges
    for node_id in list(G.nodes()):
        full_path = os.path.join(vault_path, node_id)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            continue
        # find all link targets
        matches = link_pattern.findall(content)
        for match in matches:
            target_title = match.strip()
            if not target_title:
                continue
            if target_title in note_paths:
                target_node = note_paths[target_title]
                G.add_edge(node_id, target_node)
    
    return G