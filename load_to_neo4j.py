"""
Step 9: Load entities and relationships into Neo4j.

HOW TO RUN:
    pip install neo4j
    python load_to_neo4j.py

WHAT IT DOES:
- Connects to your local Neo4j (from Docker, step 5)
- Creates a node for every entity in clean_entities.json
  (labeled :Entity, with a "name" property)
- Creates a relationship for every triple in relationships.json
  (using a generic :RELATED_TO type, with the actual relation
  phrase stored as a "type" property -- Neo4j relationship types
  can't have spaces/arbitrary text, so we keep the real phrase
  as data instead)
- Clears any existing data first, so you can re-run this safely
  as you improve extraction later
"""

import json
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"  # change this if you used a different password in step 5

ENTITIES_FILE = "clean_entities.json"
RELATIONSHIPS_FILE = "relationships.json"


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_entity(tx, name, frequency):
    tx.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.frequency = $frequency
        """,
        name=name, frequency=frequency,
    )


def create_relationship(tx, subject, relation, obj, sentence, source_file):
    tx.run(
        """
        MATCH (a:Entity {name: $subject})
        MATCH (b:Entity {name: $object})
        MERGE (a)-[r:RELATED_TO {type: $relation}]->(b)
        SET r.example_sentence = $sentence,
            r.source_file = $source_file
        """,
        subject=subject, object=obj, relation=relation,
        sentence=sentence, source_file=source_file,
    )


def main():
    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(RELATIONSHIPS_FILE, "r", encoding="utf-8") as f:
        relationships = json.load(f)

    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships from files.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("Clearing existing graph data...")
        session.execute_write(clear_database)

        print("Creating entity nodes...")
        for i, e in enumerate(entities, 1):
            session.execute_write(create_entity, e["entity"], e["frequency"])
            if i % 500 == 0:
                print(f"  {i}/{len(entities)} entities created...")

        print("Creating relationships...")
        skipped = 0
        for i, r in enumerate(relationships, 1):
            try:
                session.execute_write(
                    create_relationship,
                    r["subject"], r["relation"], r["object"],
                    r.get("sentence", ""), r.get("source_file", ""),
                )
            except Exception as e:
                skipped += 1
            if i % 100 == 0:
                print(f"  {i}/{len(relationships)} relationships processed...")

    driver.close()

    print(f"\nDone. Loaded {len(entities)} entities and {len(relationships) - skipped} relationships into Neo4j.")
    if skipped:
        print(f"  ({skipped} relationships skipped due to errors)")
    print("\nOpen http://localhost:7474 and run this query to see your graph:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")


if __name__ == "__main__":
    main()