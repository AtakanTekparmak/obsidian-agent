from kg.llm import LLM, QuestionReformat
from kg.generate_graph import KGBuildDriver, ConsistencyChecker
from kg.generate_md import generate_markdown_kb_json
from kg.generate_qa import generate_retrieval_attr_qas
from kg.generate_update import select_random_path_attrs, find_neighbor_by_edge
from kg.diff import diff_strings
from dotenv import load_dotenv
import os
import uuid
import json
from jinja2 import Template
from tqdm import tqdm
import networkx as nx

load_dotenv()

with open("restructure_0_hop.md", "r") as f:
    template_str = f.read()
template_0 = Template(template_str)

with open("restructure_1_2_hop.md", "r") as f:
    template_str = f.read()
template_1_2 = Template(template_str)

num_iter_per_graph = 3
num_qa_per_iter = 10
num_people = 5
num_entities = 5
world = "A large italian-american family originated from New Jersey. Few of the members in the family works in the family-owned Italian restaurant 'Pangorio'."
# example_world = "Two neighboring families in Morocco."

if __name__ == "__main__":

    instance_id = str(uuid.uuid4())
    # Ensure the instance ID is unique and can be used for file naming
    os.makedirs(f"instances/{instance_id}", exist_ok=True)
    llm = LLM()
    reformatter = QuestionReformat()

    driver = KGBuildDriver()
    """
    driver.gen_stubs(world, n_people=num_people, n_entities=num_entities)
    driver.edges(world)
    driver.enrich_and_verify(world)
    checker = ConsistencyChecker(driver.kg)
    problems = checker.run()
    
    if problems:
        print("❌ Consistency Issues:")
        for p in problems:
            print("  -", p)
        raise ValueError("Graph has consistency issues, cannot proceed.")
    """
    with open("sample.json", "r", encoding="utf-8") as f:
        json_data = f.read()
    driver.kg = driver.kg.from_json(json_data)

    print("✅ Graph valid. Payload:\n")
    # print(driver.kg.to_json())
    # Create a folder for instance
    output_path = f"instances/{instance_id}/graph.json"
    with open(output_path, "w") as f:
        f.write(driver.kg.to_json())
    # Pick 3 random person nodes and generate markdown files
    import random

    person_nodes = [
        n for n, d in driver.kg.g.nodes(data=True) if d.get("type").lower() == "person"
    ]
    if len(person_nodes) < num_iter_per_graph:
        print("Not enough person nodes to generate markdown files.")
    else:
        selected_nodes = random.sample(person_nodes, num_iter_per_graph)
        for node_id in tqdm(
            selected_nodes, desc="Generating markdown files and questions"
        ):

            print(f"Generating markdown for {node_id}")
            mem_id = "memory_" + uuid.uuid4().hex
            os.makedirs(f"instances/{instance_id}/{mem_id}", exist_ok=True)
            """
            md_file = generate_markdown_kb_json(driver.kg.g, node_id=node_id)
            md_file["mem_id"] = mem_id
            user_md = md_file["user_md"]

            qa = generate_retrieval_attr_qas(driver.kg.g, start=node_id)

            retrieval_questions = {"zero_hop": [], "one_hop": [], "two_hop": []}

            # 0-hop questions
            zero_hops = qa.get("zero_hop", [])
            retrieval_questions["zero_hop"] = reformatter.reformat(user=driver.kg.g.nodes[node_id]["name"],personal_info=user_md, questions=zero_hops, is_zero=True)
            print("Generated 0-hop questions")

            # 1-2-hop questions
            one_hops = qa.get("one_hop", [])[:num_qa_per_iter]
            two_hops = qa.get("two_hop", [])[:num_qa_per_iter]
            retrieval_questions["one_hop"] = reformatter.reformat(user=driver.kg.g.nodes[node_id]["name"],personal_info=user_md, questions=one_hops, is_zero=False)
            print("Generated 1-hop questions")
            retrieval_questions["two_hop"] = reformatter.reformat(user=driver.kg.g.nodes[node_id]["name"],personal_info=user_md, questions=two_hops, is_zero=False)
            print("Generated 2-hop questions")
            """
            update_queries = {"zero_hop": [], "one_hop": [], "two_hop": []}

            print("Generating updates")
            # Generate updates
            for hop in [2, 1, 0]:
                try:
                    path = select_random_path_attrs(driver.kg.g, node_id, hops=hop)
                except:
                    print(f"Failed to select random path for hop {hop}. Skipping.")
                    continue
                updated = reformatter.reformat_update(
                    user=path["path"][0], path=path
                )
                queries = updated[:2]
                data = updated[-1]
                new_graph = driver.kg.g.copy()

                if "attribute_name" in data:
                    nx.set_node_attributes(
                        new_graph,
                        {
                            node_id: {
                                data["attribute_name"]: data["attribute_value"]
                            }
                        },
                    )
                    new_md = generate_markdown_kb_json(new_graph, node_id=node_id)
                    old_md = generate_markdown_kb_json(driver.kg.g, node_id=node_id)
                    diff = diff_strings(old_md["user_md"], new_md["user_md"])

                elif "name" in data:
                    new_id = str(uuid.uuid4())
                    rel_name = path["path"][-2]
                    end_id = find_neighbor_by_edge(
                        new_graph, path["changed_node_id"], rel_name
                    )[0]
                    new_graph.remove_edge(
                        path["changed_node_id"], end_id, key=rel_name
                    )
                    if "entity_type" in data:
                        new_graph.add_node(
                            new_id,
                            name=data["name"],
                            type="Entity",
                            entity_type=data["entity_type"],
                        )
                    else:
                        new_graph.add_node(
                            new_id, name=data["name"], type="Person"
                        )

                    new_graph.add_edge(
                        path["changed_node_id"], new_id, key=rel_name
                    )
                    slug_name = (
                        driver.kg.g.nodes[path["changed_node_id"]]["name"]
                        .lower()
                        .replace(" ", "_")
                    )

                    removed_node_slug_name = (
                        driver.kg.g.nodes[end_id]["name"].lower().replace(" ", "_")
                    )
                    added_node_slug_name = data["name"].lower().replace(" ", "_")

                    new_md = generate_markdown_kb_json(new_graph, node_id=node_id)
                    old_md = generate_markdown_kb_json(driver.kg.g, node_id=node_id)
                    file_path1 = ""
                    file_path2 = ""
                    new_string = None
                    old_string = None

                    for entity_name in old_md["entities"]:
                        if entity_name["entity_name"] == slug_name:
                            old_string = entity_name["entity_file_content"]
                            new_string = [ent["entity_file_content"] for ent in new_md["entities"] if ent["entity_name"] == slug_name][0]
                            file_path1 = entity_name["entity_file_path"]
                            break
                    if new_string is None and old_string is None:
                        old_string = old_md["user_md"]
                        new_string = new_md["user_md"]
                        file_path1 = "user.md"
                    diff_1 = diff_strings(old_string, new_string)

                    for entity_name in new_md["entities"]:
                        if entity_name["entity_name"] == added_node_slug_name:
                            old_string = ""
                            new_string = [ent["entity_file_content"] for ent in new_md["entities"] if ent["entity_name"] == added_node_slug_name][0]
                            file_path2 = entity_name["entity_file_path"]
                            break
                    diff_2 = diff_strings(old_string, new_string)
                    diff = "===" + file_path1 + "===" +"\n" + diff_1 + "===" + file_path2 + "===" +"\n" + diff_2

                if hop == 0:
                    update_queries["zero_hop"].append(
                        {
                            "query": queries[0],
                            "diff": diff,
                        }
                    )
                    update_queries["zero_hop"].append(
                        {
                            "query": queries[1],
                            "diff": diff,
                        }
                    )
                elif hop == 1:
                    update_queries["one_hop"].append(
                        {
                            "query": queries[0],
                            "diff": diff,
                        }
                    )
                    update_queries["one_hop"].append(
                        {
                            "query": queries[1],
                            "diff": diff,
                        }
                    )
                elif hop == 2:
                    update_queries["two_hop"].append(
                        {
                            "query": queries[0],
                            "diff": diff,
                        }
                    )
                    update_queries["two_hop"].append(
                        {
                            "query": queries[1],
                            "diff": diff,
                        }
                    )

            # Save questions to file
            questions_path = (
                f"instances/{instance_id}/{mem_id}/retrieval_questions.json"
            )
            with open(questions_path, "w") as f:
                json.dump(retrieval_questions, f, indent=2)

            updates_path = f"instances/{instance_id}/{mem_id}/update_queries.json"
            with open(updates_path, "w") as f:
                json.dump(update_queries, f, indent=2)

            with open(f"instances/{instance_id}/{mem_id}/base_memory.json", "w") as f:
                f.write(json.dumps(md_file, indent=2))
