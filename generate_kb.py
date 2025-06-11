from data.pipeline import generate_personas, generate_kb
from training.retrieval.dataset import build_verifiers_dataset

def main():
    """Generate knowledge base with personas for Groningen, Netherlands in 2025."""
    scenario = "Groningen, the Netherlands in 2025"
    print(f"Generating knowledge base for scenario: {scenario}")
    
    # Create KB with personas and save to file
    personas = generate_personas(8, scenario, save=True)
    kb = generate_kb(personas, save=True)
    
    print(f"Knowledge base generated successfully with {len(kb.items)} personas.")

    print("Building verifiers dataset...")
    dataset = build_verifiers_dataset(kb, save=True)
    print(f"Verifiers dataset built successfully with {len(dataset)} items.")

if __name__ == "__main__":
    main() 