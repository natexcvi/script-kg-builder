import json
import re

import networkx as nx
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
        kg_df = self.__load_knowledge_graph_with_embeddings(
            "../results/12_years_a_slave.csv"
        )
        n = 1000
        kg = self.__kg_df_to_graph(kg_df)
        pair_gen = self.__build_pairs(kg)
        pairs = []
        for _ in tqdm(range(n), total=n):
            try:
                pairs.append(next(pair_gen))
            except StopIteration:
                break
        # filter out duplicate pairs, under symmetric relations
        pairs = list(set([tuple(sorted(pair)) for pair in pairs]))
        pairs.sort()
        print(pairs)
        self.texts1 = [pair[0] for pair in pairs]
        self.texts2 = [pair[1] for pair in pairs]
        self.labels = [1] * len(pairs)

        # sanity check
        # self.texts1 = ["Anne"]
        # self.texts2 = ["Solomon"]
        # self.labels = [1]
        # self.texts1.append("Solomon")
        # self.texts2.append("Alexander")
        # self.labels.append(-1)

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

    def __kg_df_to_graph(self, kg_df):
        """
        Convert a knowledge graph to a networkx graph.
        :param kg_df: the knowledge graph
        :return: the networkx graph
        """
        graph = nx.from_pandas_edgelist(
            kg_df,
            "head",
            "tail",
            ["relation", "relation_embedding"],
            create_using=nx.MultiDiGraph(),
        )
        return graph

    def __build_pairs(self, kg: nx.Graph, num_hops: int = 2):
        """
        Build pairs of entities from the knowledge graph
        based on symmetric relations. For example, (Tom, plays, football)
        and (Erica, plays, football) will result in the pair (Tom, Erica).
        In order to account for more fuzzy relation matching, we will use
        the cosine similarity between the relation embeddings to determine
        if two relations are similar enough to be considered symmetric.
        :param kg: the knowledge graph
        :param num_hops: the number of hops to consider for symmetric relations
        :yields: the pairs
        """
        # reverse the edges in the knowledge graph
        kg_rev = nx.reverse(kg, copy=True)

        for pivot in kg_rev.nodes:
            for neighbor in kg_rev.neighbors(pivot):
                for neighbor2 in kg_rev.neighbors(pivot):
                    if neighbor2 == pivot or neighbor == neighbor2:
                        continue
                    for edge in kg_rev[pivot][neighbor]:
                        for edge2 in kg_rev[pivot][neighbor2]:
                            if (
                                self.__relation_similarity(
                                    kg_rev[pivot][neighbor][edge]["relation_embedding"],
                                    kg_rev[pivot][neighbor2][edge2][
                                        "relation_embedding"
                                    ],
                                )
                                > 0.90
                            ):
                                yield (neighbor, neighbor2)

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

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        return {
            # "pixel_values": self.images[idx],
            "text1": self.texts1[idx],
            "text2": self.texts2[idx],
            "label": self.labels[idx],
        }


def evaluate_model(text_model, text_processor):
    # evaluate the model on a test set
    test_text_data_1 = [
        "Sam",
        "Anne",
        "Uncle Abram",
        "Solomon",
        "Solomon",
        "A picture of an apple",
        "A glass of water on the table",
    ]
    test_text_data_2 = [
        "Solomon",
        "Alonzo",
        "Alonzo",
        "Slaves",
        "Free man",
        "A picture of an orange",
        "An airplane in the sky",
    ]

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
        print(
            pd.DataFrame.from_dict(
                {
                    "text1": test_text_data_1,
                    "text2": test_text_data_2,
                    "similarity": similarity_scores.tolist(),
                }
            )
        )


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def __mask(self, batch_size):
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        return mask

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return torch.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = torch.nn.functional.normalize(proj_1, p=2, dim=1)
        z_j = torch.nn.functional.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = self.__mask(batch_size).to(similarity_matrix.device) * torch.exp(
            similarity_matrix / self.temperature
        )

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        return loss


def loss_fn(input1, input2, label):
    """
    Symmetric contrastive loss function.
    :param input1: the first input
    :param input2: the second input
    :param label: the label
    """
    return nn.CosineEmbeddingLoss()(input1, input2, label)


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
batch_size = 128
num_epochs = 40
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
criterion = ContrastiveLoss(batch_size)
# criterion = nn.CosineEmbeddingLoss()
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

        optimizer.zero_grad()

        text_input_1 = text_processor(batch["text1"], return_tensors="pt", padding=True)
        text_input_2 = text_processor(batch["text2"], return_tensors="pt", padding=True)
        loss = criterion(
            text_model(**text_input_1).text_embeds,
            # image_model(batch["pixel_values"]),
            text_model(**text_input_2).text_embeds,
            # batch["label"],
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

print("Fine-tuning complete! Model saved at:", output_dir)

print("Post-training:")
evaluate_model(text_model, text_processor)
