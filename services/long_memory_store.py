import os
import json
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
        self.store: Dict[Tuple[str, str], List[Tuple[str, np.ndarray]]] = {}

    def embed(self, text: str) -> np.ndarray:
        response = openai.embeddings.create(
            input=[text], 
            model=self.model,
            dimensions=256
        )
        return np.array(response.data[0].embedding)

    def put(self, namespace: Tuple[str, str], key: str, data: Dict[str, str]):
        user_id, memory = namespace
        embedding = self.embed(data["text"])

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

        new_metadata = [{"data": text} for text, _ in self.store[namespace]]
        new_embeddings = np.stack([vec for _, vec in self.store[namespace]])

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

    def load(self):
        for namespace in os.listdir(self.base_dir):
            ns_dir = os.path.join(self.base_dir, namespace)
            if not os.path.isdir(ns_dir):
                continue

            for file in os.listdir(ns_dir):
                if file.endswith("_meta.json"):
                    prefix = file.replace("_meta.json", "")
                    meta_path = os.path.join(ns_dir, f"{prefix}_meta.json")
                    vecs_path = os.path.join(ns_dir, f"{prefix}_vecs.npy")

                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                    embeddings = np.load(vecs_path)

                    key = (namespace, prefix)
                    self.store[key] = [
                        (entry["id"], entry["text"], emb)
                        for entry, emb in zip(metadata, embeddings)
                    ]


def main():
    store = MemoryStore()

    # store.put(("user", "1"), "semantic_memory", {"text": "I love pizzas"})
    # store.put(("user", "1"), "semantic_memory", {"text": "I am a engineer"})

    print(store.get(("user", "1"), "semantic_memory"))
    # results = store.search(("user", "1"), "semantic_memory", query="I'm hungry", limit=1)
    # print(results)

    # checking the shape of embedding
    # embeddings = np.load("memory_store_database/user_1/semantic_memory_vecs.npy")
    # print(embeddings.shape)
    

if __name__ == "__main__":
    main()
