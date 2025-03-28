## Memory Agent Data Generation Pipeline

### Data Generation Steps

1. **Generate Diverse Knowledge Base**
   - Generate personal stories about individuals, their relationships, and daily activities
   - Ensure knowledge includes varied temporal contexts (past events, current states, future plans)
   - Include content with inherent hierarchical and network structures
   - Balance factual content with subjective experiences and perspectives

2. **Extract Structured Facts and Relationships**
   - Generate chronological timelines of events from narratives
   - Extract entity information (people, places, objects, concepts)
   - Document properties and attributes of each entity with timestamps
   - Map explicit and implicit relationships between entities
   - Include metadata for each fact (timestamp, source, certainty, importance)
   - Incorporate state changes in facts over time (e.g., "[2023-04-15] Alex is married" → "[2023-10-22] Alex got divorced")
   - Distinguish between objective facts and subjective interpretations
   - Generate non-chronological facts with timestamps (characteristics, medical conditions, principles)

3. **Generate Knowledge Graph Structure (Folders and Files)**
   - Design a coherent ontology for folder organization (entities, events, concepts)
   - Create hierarchical folder structure with clear organizational logic
   - Implement consistent file naming conventions that enhance discoverability
   - Populate files with extracted facts, narratives, and relevant timestamps
   - Create index files that provide overview of content categories
   - Allow for organic linking between related content without enforcing bidirectionality
   - Include metadata headers in files (creation date, related entities, importance)
   - Generate initial links based on direct relationships in the data

4. **Generate Retrieval-Focused Questions and Answers**
   - Create simple fact retrieval questions that target specific information
   - Develop questions that test navigation of entity relationships
   - Generate temporal questions about information changes over time
   - Design multi-hop reasoning questions requiring graph traversal
   - Include questions about contradictions or information updates
   - Create questions requiring aggregation of information from multiple files
   - Ensure questions test understanding of the knowledge graph structure
   - Generate paired answers with reference paths through the knowledge graph

5. **Generate Natural Multi-turn Conversations with Implicit Information**
   - Create realistic conversation flows without explicit knowledge-sharing intent
   - Include casual mentions of life events that imply state changes ("Just got back from the divorce court, glad all of this is over")
   - Incorporate subtle references to health changes ("These new pills are making me feel much better")
   - Design conversations that contain contradictions or updates to previously established facts
   - Include contextual clues that require inference to properly update the knowledge graph
   - Simulate natural topic transitions that test the agent's ability to track and organize information
   - Vary information explicitness levels (highly implicit for personal matters, somewhat more direct for medical information)
   - Generate expected knowledge graph updates that should result from each conversation

6. **Evaluation Framework: Graph Structure and Content Quality**
   - **Structural Metrics**
     - Graph connectivity (largest_weakly_connected_fraction)
     - Community structure (directed_modularity)
     - Node distribution (degree_distribution_heterogeneity)
     - Local interconnectedness (directed_clustering_coefficient)
     - Information bridge quality (directed_betweenness_centrality)
   
   - **Content Quality Metrics**
     - Note cohesion (note_cohesion)
     - Folder organization quality (folder_cohesion)
     - Information uniqueness (uniqueness)
     - Linking relevance (link_relevance)