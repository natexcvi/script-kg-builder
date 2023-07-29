import json
import re

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)


class CustomDataset(Dataset):
    def __init__(self, images, texts1, texts2, labels):
        self.images = images
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        kg = self.__load_knowledge_graph_with_embeddings(
            "../results/12_years_a_slave.csv"
        )
        self.pairs = list(tqdm(self.__build_pairs(kg), total=len(kg)))
        self.texts1 = [pair[0] for pair in self.pairs]
        self.texts2 = [pair[1] for pair in self.pairs]

    def __load_knowledge_graph_with_embeddings(self, path: str):
        """
        Load th knowledge grpah from a CSV with columns (head, relation, relation_embedding, tail).
        :param path: the path to the CSV file
        :return: the knowledge graph
        """

        df = pd.read_csv(path)
        df["relation_embedding"] = df["relation_embedding"].apply(
            lambda x: json.loads(x)
        )
        return df

    def __relation_similarity(self, relation1, relation2):
        """
        Calculate the cosine similarity between two relation embeddings.
        :param relation1: the first relation embedding
        :param relation2: the second relation embedding
        :return: the cosine similarity
        """
        return np.dot(relation1, relation2) / (
            np.linalg.norm(relation1) * np.linalg.norm(relation2)
        )

    def __build_pairs(self, kg_df):
        """
        Build pairs of entities from the knowledge graph
        based on symmetric relations. For example, (Tom, plays, football)
        and (Erica, plays, football) will result in the pair (Tom, Erica).
        In order to account for more fuzzy relation matching, we will use
        the cosine similarity between the relation embeddings to determine
        if two relations are similar enough to be considered symmetric.

        :param kg_df: the knowledge graph
        :return: the pairs
        """
        for i, row in kg_df.iterrows():
            for j, row2 in kg_df.iterrows():
                if i == j:
                    continue
                if (
                    self.__relation_similarity(
                        row["relation_embedding"], row2["relation_embedding"]
                    )
                    > 0.90
                ):
                    yield (row["head"], row2["head"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "pixel_values": self.images[idx],
            "text1": self.texts1[idx],
            "text2": self.texts2[idx],
            "label": self.labels[idx],
        }


def evaluate_model(text_model, text_processor):
    # evaluate the model on a test set
    test_text_data_1 = [
        "test",
        "not a test",
        "test",
        "not test",
        "water",
    ]
    test_text_data_2 = [
        "experiment",
        "the real deal",
        "the real deal",
        "experiment",
        "apple",
    ]

    test_image_data = np.array([[1, 1, 1], [1, 1, 1]])

    test_dataset = CustomDataset(
        test_image_data, test_text_data_1, test_text_data_2, [1, 1]
    )

    text_model.eval()
    with torch.no_grad():
        # calculate the cosine similarity between the two text embeddings
        text_input_1 = text_processor(
            test_text_data_1, return_tensors="pt", padding=True
        )
        text_input_2 = text_processor(
            test_text_data_2, return_tensors="pt", padding=True
        )
        text_embedding_1 = text_model(**text_input_1).text_embeds
        text_embedding_2 = text_model(**text_input_2).text_embeds
        similarity_scores = nn.CosineSimilarity(dim=1)(
            text_embedding_1, text_embedding_2
        )
        print(similarity_scores)


clip_model_name = "openai/clip-vit-base-patch16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLIP model and processor
text_model = CLIPTextModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
text_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image_model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32"
)
image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained(clip_model_name)

print("Pre-training:")
evaluate_model(text_model, text_processor)

# Fine-tuning parameters
batch_size = 8
num_epochs = 10
learning_rate = 1e-5

# Create the dataset and data loader
image_data = np.array([[1, 1, 1], [1, 1, 1]])
text_data_1 = np.array(["test", "not a test", "test", "not test"])
text_data_2 = np.array(
    [
        "experiment",
        "the real deal",
        "the real deal",
        "experiment",
    ]
)
dataset = CustomDataset(image_data, text_data_1, text_data_2, [1, 1, -1, -1])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.AdamW(text_model.parameters(), lr=learning_rate)

# Fine-tuning loop
text_model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in data_loader:
        # inputs = processor(
        #     batch["pixel_values"], batch["text"], return_tensors="pt", padding=True
        # )
        # inputs = {k: v.to(device) for k, v in inputs.items()}

        # # The label for CosineEmbeddingLoss is always 1 for positive pairs
        # labels = torch.ones(inputs["input_ids"].shape[0]).to(device)

        optimizer.zero_grad()

        # outputs = clip_model(**inputs, labels=labels)
        # loss = outputs.loss
        # loss.backward()
        text_input_1 = text_processor(batch["text1"], return_tensors="pt", padding=True)
        text_input_2 = text_processor(batch["text2"], return_tensors="pt", padding=True)
        loss = criterion(
            text_model(**text_input_1).text_embeds,
            # image_model(batch["pixel_values"]),
            text_model(**text_input_2).text_embeds,
            batch["label"],
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss}")

# Save the fine-tuned model
output_dir = "./fine_tuned_clip_model"
text_model.save_pretrained(output_dir)
image_model.save_pretrained(output_dir)
# clip_model.save_pretrained(output_dir)
# processor.save_pretrained(output_dir)

print("Fine-tuning complete! Model saved at:", output_dir)

print("Post-training:")
evaluate_model(text_model, text_processor)
