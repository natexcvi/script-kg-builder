import itertools
import json
from cgitb import text

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoProcessor, AutoTokenizer,
                          CLIPTextModelWithProjection,
                          CLIPVisionModelWithProjection)


class KGDataset(Dataset):
    def __init__(self, kg_csv_path: str, max_pairs: int = 1000, max_hops: int = 3):
        """
        Create a dataset of pairs of entities from a knowledge graph.

        Parameters:
        ----------
        kg_csv_path : the path to the CSV file containing the knowledge graph
        max_pairs : the maximum number of pairs to generate
        """

        kg_df = self.__load_knowledge_graph_with_embeddings(kg_csv_path)
        kg = self.__kg_df_to_graph(kg_df)
        pair_gen = self.__build_pairs(kg, num_hops=max_hops)
        pairs = []
        for _ in tqdm(range(max_pairs), total=max_pairs):
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

        Parameters:
        ----------
        path : the path to the CSV file

        Returns:
        -------
        df : the knowledge graph as a pandas dataframe
        """

        df = pd.read_csv(path)
        df["relation_embedding"] = df["relation_embedding"].apply(json.loads)
        return df

    def __kg_df_to_graph(self, kg_df):
        """
        Convert a knowledge graph to a networkx graph.

        Parameters:
        ----------
        kg_df : the knowledge graph as a pandas dataframe

        Returns:
        -------
        graph : the knowledge graph as a networkx graph
        """
        graph = nx.from_pandas_edgelist(
            kg_df,
            "head",
            "tail",
            ["relation", "relation_embedding"],
            create_using=nx.MultiDiGraph(),
        )
        return graph

    def __build_pairs(self, kg: nx.Graph, num_hops: int = 3):
        """
        Build pairs of entities from the knowledge graph
        based on symmetric relations. For example, (Tom, plays, football)
        and (Erica, plays, football) will result in the pair (Tom, Erica).
        In order to account for more fuzzy relation matching, we will use
        the cosine similarity between the relation embeddings to determine
        if two relations are similar enough to be considered symmetric.

        Parameters:
        ----------
        kg : the knowledge graph
        num_hops : the number of hops to consider for symmetric relations

        Yields:
        ------
        pairs : the pairs of entities
        """
        kg = nx.reverse_view(kg)
        for pivot in kg.nodes:
            pairs = {0: [(pivot, pivot)]}
            for k in range(1, num_hops + 1):
                if k not in pairs:
                    pairs[k] = []
                for u, v in pairs[k - 1]:
                    for ux, uy in itertools.product(
                        kg.edges(u, data=True), kg.edges(v, data=True)
                    ):
                        _, x, ux_data = ux
                        _, y, vy_data = uy
                        if x == y:
                            continue
                        if (x, y) in pairs[k - 1]:
                            continue
                        if (
                            self.__relation_similarity(
                                ux_data["relation_embedding"],
                                vy_data["relation_embedding"],
                            )
                            >= 0.90
                        ):
                            pairs[k].append((x, y))
                            yield (x, y)

    def __relation_similarity(self, relation1, relation2):
        """
        Calculate the cosine similarity between two relation embeddings.

        Parameters:
        ----------
        relation1 : the first relation embedding
        relation2 : the second relation embedding

        Returns:
        -------
        similarity : the cosine similarity between the two relation embeddings
        """
        return np.dot(relation1, relation2) / (
            np.linalg.norm(relation1) * np.linalg.norm(relation2)
        )

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        return {
            "text1": self.texts1[idx],
            "text2": self.texts2[idx],
            "label": self.labels[idx],
        }


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper

    [Source](https://theaisummer.com/simclr/)
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


class MultiModalKGCLIP(nn.Module):
    def __init__(self, batch_size=128, num_epochs=40, learning_rate=1e-5):
        super().__init__()
        (
            self.text_model,
            self.text_processor,
            self.image_model,
            self.image_processor,
        ) = self.__get_models()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    @staticmethod
    def __get_models():
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
        return (
            text_model,
            text_processor,
            image_model,
            image_processor,
        )

    def forward(self, text_input, image_input):
        text_embedding = self.text_model(**text_input).text_embeds
        image_embedding = self.image_model(**image_input).image_embeds
        return text_embedding, image_embedding

    def save_pretrained(self, output_dir):
        self.text_model.save_pretrained(output_dir)
        self.image_model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, pretrained_dir):
        text_model = CLIPTextModelWithProjection.from_pretrained(pretrained_dir)
        image_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_dir)
        return cls(text_model, image_model)

    def train(self, mode: bool = True):
        self.text_model.train(mode)
        self.image_model.train(mode)
        return super().train(mode)

    def eval(self):
        self.text_model.eval()
        self.image_model.eval()
        return super().eval()

    def fit(self, data_loader: DataLoader):
        criterion = ContrastiveLoss(self.batch_size)
        optimizer = optim.AdamW(self.text_model.parameters(), lr=self.learning_rate)
        # Fine-tuning loop
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch in data_loader:
                optimizer.zero_grad()

                text_input_1 = self.text_processor(
                    batch["text1"], return_tensors="pt", padding=True
                )
                text_input_2 = self.text_processor(
                    batch["text2"], return_tensors="pt", padding=True
                )
                loss = criterion(
                    self.text_model(**text_input_1).text_embeds,
                    self.text_model(**text_input_2).text_embeds,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {avg_loss}")

    def predict(self, text_data):
        self.eval()
        with torch.no_grad():
            # get the image embedding
            text_input = self.text_processor(
                text_data, return_tensors="pt", padding=True
            )
            text_embedding = self.text_model(**text_input).text_embeds
            return text_embedding

    def evaluate(self, text_data_1, text_data_2):
        self.eval()
        with torch.no_grad():
            # calculate the cosine similarity between the two text embeddings
            text_embedding_1 = self.predict(text_data_1)
            text_embedding_2 = self.predict(text_data_2)
            similarity_scores = nn.CosineSimilarity(dim=1)(
                text_embedding_1, text_embedding_2
            )
            print(
                pd.DataFrame.from_dict(
                    {
                        "text1": text_data_1,
                        "text2": text_data_2,
                        "similarity": similarity_scores.tolist(),
                    }
                )
            )


def plot_embeddings(model, text_data):
    tsne = TSNE(n_components=2, random_state=0, metric="cosine", perplexity=5)
    visualisation_set = text_data
    text_embeddings = model.predict(visualisation_set).cpu().numpy()
    text_embeddings = tsne.fit_transform(text_embeddings)
    text_embeddings = pd.DataFrame(
        np.hstack((text_embeddings, np.array(visualisation_set).reshape(-1, 1))),
        columns=["x", "y", "text"],
    )
    text_embeddings["x"] = text_embeddings["x"].astype(float)
    text_embeddings["y"] = text_embeddings["y"].astype(float)
    fig = px.scatter(text_embeddings, x="x", y="y", text="text")
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=800,
        title_text="Embeddings of several words",
        title_x=0.5,
        title_y=0.9,
        title_font_size=30,
    )
    fig.show()


if __name__ == "__main__":
    batch_size = 128

    model = MultiModalKGCLIP(batch_size=batch_size, num_epochs=40)

    eval_1 = [
        "Sam",
        "Solomon",
        "Anne",
        "Anne",
        "Uncle Abram",
        "Solomon",
        "Solomon",
        "Edwin Epps",
        "William Ford",
        "A picture of an apple",
        "A glass of water on the table",
    ]
    eval_2 = [
        "Solomon",
        "Anne",
        "Alonzo",
        "Margaret",
        "Alonzo",
        "Slaves",
        "Free man",
        "Solomon",
        "Solomon",
        "A picture of an orange",
        "An airplane in the sky",
    ]

    print("Pre-training:")
    model.evaluate(eval_1, eval_2)
    plot_embeddings(model, list(set(eval_1 + eval_2)))
    dataset = KGDataset("../results/12_years_a_slave.csv", max_pairs=1000)
    model.fit(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    print("Post-training:")
    model.evaluate(eval_1, eval_2)
    output_dir = "./fine_tuned_clip_model"
    model.save_pretrained(output_dir)
    print(f"Saved model to '{output_dir}'")
    plot_embeddings(model, list(set(eval_1 + eval_2)))
