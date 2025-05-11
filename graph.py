import os
import random
import string
import networkx as nx
from training.obsidian.utils import build_graph_from_vault, visualize_obsidian_graph

# Function to get basic graph statistics
def print_graph_stats(G):
    """
    Prints basic statistics about the graph
    
    Parameters:
    -----------
    G : nx.DiGraph
        The graph to analyze
    """
    print(f"Number of notes: {G.number_of_nodes()}")
    print(f"Number of links: {G.number_of_edges()}")
    
    # Get most linked notes (highest in-degree)
    in_degrees = dict(G.in_degree())
    most_linked = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nMost referenced notes:")
    for node, count in most_linked:
        title = G.nodes[node].get('title', os.path.basename(node).replace('.md', ''))
        print(f"  - {title}: {count} references")
    
    # Get notes with most outgoing links
    out_degrees = dict(G.out_degree())
    most_outgoing = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nNotes with most outgoing links:")
    for node, count in most_outgoing:
        title = G.nodes[node].get('title', os.path.basename(node).replace('.md', ''))
        print(f"  - {title}: {count} links")
    
    # Identify unconnected notes
    isolated = list(nx.isolates(G))
    print(f"\nNumber of isolated notes: {len(isolated)}")
    
    # Calculate connected components
    components = list(nx.weakly_connected_components(G))
    print(f"Number of connected components: {len(components)}")
    print(f"Size of largest connected component: {len(max(components, key=len))}")

def create_mock_obsidian_vault(base_path, num_folders=5, num_notes_per_folder=5, 
                               num_standalone_notes=10, link_density=0.3):
    """
    Creates a mock Obsidian vault structure with folders, notes, and internal links.
    
    Parameters:
    -----------
    base_path : str
        The directory path where the mock vault will be created
    num_folders : int
        Number of folders to create
    num_notes_per_folder : int
        Number of notes to create in each folder
    num_standalone_notes : int
        Number of notes to create in the root directory
    link_density : float
        Probability (0-1) of creating a link between any two notes
        
    Returns:
    --------
    dict
        Dictionary mapping note titles to their paths
    """
    # Create base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    note_paths = {}  # Maps note titles to their paths
    all_notes = []   # List of all note titles
    
    # Create some themes for the notes
    themes = ["Projects", "Ideas", "Research", "Books", "People", 
              "Meetings", "Goals", "Journal", "Reference", "Concepts"]
    
    # Create some common words for note content
    common_words = ["knowledge", "information", "idea", "concept", "theory", 
                    "project", "task", "goal", "research", "development",
                    "learning", "thinking", "writing", "reading", "planning"]
    
    # Function to generate a note title
    def generate_title():
        theme = random.choice(themes)
        suffix = ''.join(random.choices(string.ascii_uppercase, k=3))
        return f"{theme}-{suffix}"
    
    # Function to generate note content with links
    def generate_content(title, all_notes, link_density):
        paragraphs = []
        
        # Title section
        paragraphs.append(f"# {title}\n\n")
        
        # Add a few paragraphs of text
        for _ in range(random.randint(2, 5)):
            para_words = random.randint(30, 100)
            paragraph = ' '.join(random.choices(common_words, k=para_words))
            paragraphs.append(paragraph + "\n\n")
        
        # Add links to other notes
        links_section = []
        other_notes = [note for note in all_notes if note != title]
        
        if other_notes:
            for note in other_notes:
                if random.random() < link_density:
                    links_section.append(f"- [[{note}]]\n")
        
        if links_section:
            paragraphs.append("## Related Notes\n\n")
            paragraphs.extend(links_section)
        
        return ''.join(paragraphs)
    
    # Create standalone notes in the root directory
    for _ in range(num_standalone_notes):
        note_title = generate_title()
        while note_title in note_paths:  # Ensure unique titles
            note_title = generate_title()
        
        note_path = os.path.join(base_path, f"{note_title}.md")
        note_paths[note_title] = note_path
        all_notes.append(note_title)
    
    # Create folders and notes within them
    for i in range(num_folders):
        folder_name = f"Folder-{i+1}"
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create notes in this folder
        for _ in range(num_notes_per_folder):
            note_title = generate_title()
            while note_title in note_paths:  # Ensure unique titles
                note_title = generate_title()
            
            note_path = os.path.join(folder_path, f"{note_title}.md")
            note_paths[note_title] = note_path
            all_notes.append(note_title)
    
    # Now write content to all the notes with links
    for note_title, note_path in note_paths.items():
        content = generate_content(note_title, all_notes, link_density)
        with open(note_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created mock Obsidian vault at {base_path} with {len(all_notes)} notes")
    return note_paths

def main():
    # Set up paths (save to current directory/mock_obsidian_vault)
    base_dir = os.path.join(os.path.dirname(__file__), "mock_obsidian_vault")
    
    # 1. Create a mock Obsidian vault
    note_paths = create_mock_obsidian_vault(
        base_dir,
        num_folders=4,
        num_notes_per_folder=6,
        num_standalone_notes=8,
        link_density=0.2
    )
    
    # 2. Build the graph from the vault
    G = build_graph_from_vault(base_dir)
    
    # 3. Print some statistics about the graph
    print_graph_stats(G)
    
    # 4. Visualize the graph
    # By folder colors
    visualize_obsidian_graph(
        G,
        output_path=os.path.join(base_dir, "graph_by_folder.png"),
        node_color_by='folder',
        layout_type='spring'
    )
    
    # By connection count
    visualize_obsidian_graph(
        G,
        output_path=os.path.join(base_dir, "graph_by_connections.png"),
        node_color_by='connections',
        layout_type='kamada_kawai'
    )
    
    print("Done! Check the output images in your mock vault directory.")

if __name__ == "__main__":
    main()