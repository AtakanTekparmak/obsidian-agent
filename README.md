Memory Agent (Using Obsidian)

Stage 1:
 * Train an LLM agent to use .md files and folders, both with size limits, like Obsidian
 * Agent would use Pythonic function calling as explained in [DPAB-a](https://huggingface.co/blog/andthattoo/dpab-a) and [Dria-Agent-a](https://huggingface.co/blog/andthattoo/dria-agent-a).
 * Agent would have access to methods:
    * `create_file(file_path: str, content: str = "")`: Creates a new .md file in the specified path with the provided content.
    * `create_dir(dir_path: str)`: Creates a new directory in the specified path.
    * `get_size(file_or_dir_path: str)`: Returns the size of a file or directory in bytes. If left empty, returns the total size of all files and directories in the memory.
    * `write_to_file(file_path: str, content: str)`: Writes to a file in the specified path with the provided content.
    * `read_file(file_path: str)`: Reads the content of a file in the specified path.
    * `list_files(dir_path: Optional[str] = None)`: Lists all files and directories in the specified path, or the entire memory if left empty.
    * `delete_file(file_path: str)`: Deletes a file in the specified path.
    * `go_to_link(link_string: str)`: Goes to a link (located in a note).
 * The method `create_file` will check the size constaints of the files and folders and will throw an error if the constraints are exceeded. (Or should we instead make it a reward also?)
 * Agent would be trained to generate an obsidian vault given a text content
 * Allow referencing of other files (like Obsidian)
 * Agent would be trained in two stages, Stage 1.1 and Stage 1.2
 * The following metrics could be used to evaluate the agent:
    * **Stage 1 Metrics (Structural):**
        * **Directed Modularity:** Measures how well the vault is partitioned into densely connected communities. High modularity indicates distinct topical clusters.
        * **Directed Clustering Coefficient:** Captures the prevalence of closed triads in the vault, reflecting how interconnected each note's neighborhood is.
        * **Directed Betweenness Centrality:** Encourages "bridge" or "hub" notes that improve navigability between different parts of the vault.
        * **Largest Weakly Connected Fraction:** Measures overall vault connectivity. A high value means most/all notes are connected.
        * **Degree Distribution Heterogeneity:** Rewards having a skewed distribution of links, indicating hub-like structures rather than uniform connections.
    * **Stage 2 Metrics (Semantic - Added to Stage 1):**
        * **Note Cohesion:** Ensures each note is topically consistent internally. Higher cohesion means a note focuses on one main topic.
        * **Link Relevance:** Encourages links to connect semantically related notes. High relevance means linked notes are thematically similar.
        * **Folder Cohesion:** Rewards grouping semantically similar notes in the same folder. Measures thematic consistency within directories.
        * **Uniqueness:** Penalizes near-duplicate notes to avoid redundancy and improve retrieval effectiveness.
        
    Each metric outputs a value in [0,1], with the final reward being the average of relevant metrics, guiding the agent to produce a vault that is both structurally organized and semantically coherent.

Stage 2:
 * Give agent retrieved facts
 * Give different prompt (or modify it + odd special token)
 * Train on how well it retrieves info (on stage 1 model)
 * In inference
   * Have storage/retrieval mode (if differen prompt is used for stage 2)
   * Have unified prompt and inference (if a unified prompt is used for stages 1 and 2)
