import os
import json
import re
import numpy as np
import openai
from typing import Tuple, Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

class MemoryStore:
    def __init__(self, base_dir="/home/ashu/Projects/persistent-memory-agent/memory_store_database", model="text-embedding-3-small"):
        self.model = model
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.store: Dict[Tuple[str, str], List[Tuple[List[str], List[np.ndarray]]]] = {}

    def embed(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            input=[text], 
            model=self.model,
            dimensions=256
        )
        return np.array(response.data[0].embedding)

    def put(self, namespace: Tuple[str, str], key: str, data: Dict[str, List[str]]):
        user_id, memory = namespace
        embedding = []
        for item in data['text']:
            embedding.append(self.embed(item))

        if namespace not in self.store:
            self.store[namespace] = []
        self.store[namespace] = [(data["text"], embedding)]
        self._save_key(namespace, key)

    def get(self, namespace: Tuple[str, str], key: str):
        user, id = namespace
        file_prefix = self._key_to_path(f"{user}_{id}", key)
        meta_path = f"{file_prefix}_meta.json"
        vecs_path = f"{file_prefix}_vecs.npy"

        if not os.path.exists(meta_path) or not os.path.exists(vecs_path):
            return []

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        return [item["data"] for item in metadata]


    def search(self, namespace: Tuple[str, str], key: str, query: str, limit: int = 1) -> List[Dict[str, str]]:
        user, id = namespace
        file_prefix = self._key_to_path(f"{user}_{id}", key)
        meta_path = f"{file_prefix}_meta.json"
        vecs_path = f"{file_prefix}_vecs.npy"

        if not os.path.exists(meta_path) or not os.path.exists(vecs_path):
            return []

        with open(meta_path, "r") as f:
            metadata = json.load(f)
        embeddings = np.load(vecs_path)

        query_embedding = self.embed(query)

        similarities = [
            (cosine_similarity([query_embedding], [embedding])[0][0], idx, item["data"])
            for idx, (item, embedding) in enumerate(zip(metadata, embeddings))
        ]
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [{"id": str(idx), "text": text, "score": sim} for sim, idx, text in similarities[:limit]]

    def _key_to_path(self, path: str, name: str) -> str:
        dir_path = os.path.join(self.base_dir, path)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, name)

    def _save_key(self, namespace: Tuple[str, str], key: str):
        user_id, memory = namespace
        file_prefix = self._key_to_path(f"{user_id}_{memory}", key)
        meta_path = f"{file_prefix}_meta.json"
        vecs_path = f"{file_prefix}_vecs.npy"

        new_metadata = [{"data": text} for text in self.store[namespace][0][0]]
        new_embeddings = np.stack([vec for vec in self.store[namespace][0][1]])

        if os.path.exists(meta_path) and os.path.exists(vecs_path):
            with open(meta_path, "r") as f:
                existing_metadata = json.load(f)
            existing_embeddings = np.load(vecs_path)

            combined_metadata = existing_metadata + new_metadata
            combined_embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
        else:
            combined_metadata = new_metadata
            combined_embeddings = new_embeddings

        with open(meta_path, "w") as f:
            json.dump(combined_metadata, f)

        np.save(vecs_path, combined_embeddings)

    from sklearn.metrics.pairwise import cosine_similarity

    def cosine_similarity_custom(self, vec1, vec2):
        vec1 = np.array(vec1).reshape(1, -1)
        vec2 = np.array(vec2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def delete(self, namespace: Tuple[str, str], key: str, entities: List[str]):
        user_id, memory = namespace
        file_prefix = self._key_to_path(f"{user_id}_{memory}", key)
        meta_path = f"{file_prefix}_meta.json"
        vecs_path = f"{file_prefix}_vecs.npy"

        for entity in entities:
            original_entity = entity
            regex_match = re.search(r".*:(.*)", entity)
            entity = regex_match.group(1).strip() if regex_match else entity.strip()

            if os.path.exists(meta_path) and os.path.exists(vecs_path):
                with open(meta_path, "r") as f:
                    existing_metadata = json.load(f)
                existing_embeddings = np.load(vecs_path)

                assert len(existing_metadata) == len(existing_embeddings), "Metadata and embeddings length mismatch"

                new_metadata = []
                new_embeddings = []

                matched = False
                entity_embedding = None
                best_sim = -1
                best_idx = -1

                col = existing_embeddings.shape[1]

                existing_embeddings = list(existing_embeddings)

                for idx, item in enumerate(existing_metadata):
                    data_value = item["data"]
                    match = re.search(r".*:(.*)", data_value)
                    value_after_colon = match.group(1).strip() if match else data_value.strip()

                    # Case-insensitive exact match
                    if entity.lower() == value_after_colon.lower():
                        matched = True
                        continue  # Skip this item (i.e., delete it)
                    
                    # Fallback: compute cosine similarity
                    if not matched:
                        if entity_embedding is None:
                            entity_embedding = self.embed(original_entity)
                        item_embedding = existing_embeddings[idx]
                        sim = self.cosine_similarity_custom(entity_embedding, item_embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_idx = idx

                    new_metadata.append(item)
                    new_embeddings.append(existing_embeddings[idx])

                if not matched and best_sim > 0.6:
                    print(f"Removing best cosine match for '{original_entity}' with sim={best_sim:.2f}")
                    del existing_metadata[best_idx]
                    del existing_embeddings[best_idx]
                                    # Save updated files
                    with open(meta_path, "w") as f:
                        json.dump(existing_metadata, f, indent=2)

                    np.save(vecs_path, np.stack(existing_embeddings) if existing_embeddings else np.empty((0, col)))
                    continue

                # Save updated files
                with open(meta_path, "w") as f:
                    json.dump(new_metadata, f, indent=2)

                np.save(vecs_path, np.stack(new_embeddings) if new_embeddings else np.empty((0, existing_embeddings.shape[1])))
            else:
                print(f"Metadata or embedding file not found at {meta_path} / {vecs_path}")

def main():
    store = MemoryStore()

    store.put(("user", "11"), "semantic_memory", {"text": ['Name: Ashutosh','Occupation: AI Engineer', 'Food Preference: pizzas']})
    # store.put(("user", "11"), "semantic_memory", {"text": ["Occupation: engineer"]})

    # print(store.get(("user", "11"), "semantic_memory"))
    # results = store.search(("user", "1"), "semantic_memory", query="I'm hungry", limit=1)
    # print(results)

    # embeddings = np.load("memory_store_database/user_11/semantic_memory_vecs.npy")
    # print(embeddings.shape)
    # store.delete(("user", "11"), "semantic_memory", ['Occupation: AI Engineer'])

    # embeddings = np.load("memory_store_database/user_11/semantic_memory_vecs.npy")
    # print(embeddings.shape)


if __name__ == "__main__":
    main()
