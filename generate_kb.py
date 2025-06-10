#!/usr/bin/env python3

from training.retrieval import create_kb_with_personas

def main():
    """Generate knowledge base with personas for Groningen, Netherlands in 2025."""
    scenario = "Groningen, the Netherlands in 2025"
    print(f"Generating knowledge base for scenario: {scenario}")
    
    # Create KB with personas and save to file
    kb = create_kb_with_personas(num_personas=8, scenario=scenario, save=True)
    
    print(f"Knowledge base generated successfully with {len(kb.items)} personas.")

if __name__ == "__main__":
    main() 