import itertools
import json
import os
from calendar import c
from typing import Optional, Sized

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE
from thefuzz import fuzz
from thefuzz import process as fuzz_process
from torch import nn, optim
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from preprocessing import process_image


class KGDatasetBatchSampler(Sampler):
    def __init__(self, data_source: "KGDataset", batch_size: int) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        text_text = torch.randperm(len(self.data_source.text_text_pairs)).tolist()
        text_image = (
            torch.randperm(len(self.data_source.text_image_pairs))
            + len(self.data_source.text_text_pairs)
        ).tolist()
        image_text = (
            torch.randperm(len(self.data_source.image_text_pairs))
            + len(self.data_source.text_text_pairs)
            + len(self.data_source.text_image_pairs)
        ).tolist()
        image_image = (
            torch.randperm(len(self.data_source.image_image_pairs))
            + len(self.data_source.text_text_pairs)
            + len(self.data_source.text_image_pairs)
            + len(self.data_source.image_text_pairs)
        ).tolist()
        batches = (
            self.__to_chuncks(text_text)
            + self.__to_chuncks(text_image)
            + self.__to_chuncks(image_text)
            + self.__to_chuncks(image_image)
        )
        np.random.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __to_chuncks(self, lst):
        chuncks = []
        for i in range(0, len(lst), self.batch_size):
            chuncks.append(lst[i : i + self.batch_size])
        return chuncks

    def __len__(self):
        return len(self.batches)


class KGDataset(Dataset):
    def __init__(
        self,
        kg_csv_path: str,
        face_images_dir: Optional[str] = None,
        max_pairs: int = 1000,
        max_hops: int = 3,
    ):
        """
        Create a dataset of pairs of entities from a knowledge graph.

        Parameters:
        ----------
        kg_csv_path : the path to the CSV file containing the knowledge graph
        face_images_dir : the path to the directory containing the face images of film characters
        max_pairs : the maximum number of pairs to generate
        max_hops : the maximum number of hops to consider for symmetric relations
        """

        kg_df = self.__load_knowledge_graph_with_embeddings(kg_csv_path)
        kg = self.__kg_df_to_graph(kg_df)
        pair_gen = self.__build_pairs(kg, num_hops=max_hops)
        textual_pairs = []
        for _ in tqdm(range(max_pairs), total=max_pairs):
            try:
                textual_pairs.append(next(pair_gen))
            except StopIteration:
                break

        # filter out duplicate pairs, under symmetric relations
        textual_pairs = list({tuple(sorted(pair)) for pair in textual_pairs})
        textual_pairs.sort()
        print(textual_pairs)

        self.face_images = (
            self.__load_face_images(face_images_dir)
            if face_images_dir is not None
            else {}
        )
        unique_entity_names = {
            entity
            for entity in [pair[0] for pair in textual_pairs]
            + [pair[1] for pair in textual_pairs]
        }
        matched_face_images = self.match_face_images(list(unique_entity_names))
        self.__build_multimodal_pairs(textual_pairs, matched_face_images)

    def __build_multimodal_pairs(
        self, pairs: list[tuple[str, str]], matched_face_images: dict[str, list]
    ):
        """
        Build pairs of entities from the knowledge graph
        and their corresponding face images.

        Parameters:
        ----------
        pairs : the pairs of entities
        matched_face_images : the face images of the entities
        """

        self.texts1 = []
        self.texts2 = []
        self.images1 = []
        self.images2 = []
        for head, tail in pairs:
            self.texts1.append(head)
            self.texts2.append(tail)
            head_image, tail_image = (
                matched_face_images.get(head, [None])[0],
                matched_face_images.get(tail, [None])[0],
            )
            self.images1.append(head_image)
            self.images2.append(tail_image)

        self.text_text_pairs = list(zip(self.texts1, self.texts2))
        self.text_image_pairs = [
            (text, image)
            for text, image in zip(self.texts1, self.images2)
            if image is not None
        ]
        self.image_text_pairs = [
            (image, text)
            for image, text in zip(self.images1, self.texts2)
            if image is not None
        ]
        self.image_image_pairs = [
            pair for pair in zip(self.images1, self.images2) if None not in pair
        ]
        self.labels = [1] * (
            len(self.text_text_pairs)
            + len(self.text_image_pairs)
            + len(self.image_text_pairs)
            + len(self.image_image_pairs)
        )

    def match_face_images(self, names: list[str]) -> dict[str, list]:
        """
        Match the face images to the names of the characters
        using fuzzy string matching.
        If a name is not matched, then the name will not be included
        in the returned dictionary.

        Parameters:
        ----------
        names : the names of the characters

        Returns:
        -------
        images : a dictionary mapping the name of the character to a list of face images
        """
        canonical_names = list(self.face_images.keys())
        images = {}
        for name in names:
            match = fuzz_process.extractOne(name, canonical_names, score_cutoff=90)
            if match is not None:
                match, _ = match
                images[name] = self.face_images[match]
        return images

    def __load_face_images(self, path: str):
        """
        Load the face images from a directory.

        Parameters:
        ----------
        path : the path to the directory containing the face images, where
                the name of each subdirectory is the name of the character.

        Returns:
        -------
        images : a dictionary mapping the name of the character to a list of face images
        """

        images: dict[str, list] = {}
        for sub_dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, sub_dir)):
                continue
            for image_path in os.listdir(os.path.join(path, sub_dir)):
                image = process_image(os.path.join(path, sub_dir, image_path))
                if image is not None:
                    name = sub_dir
                    if name not in images:
                        images[name] = []
                    images[name].append(image)
        return images

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
        pairs : tuples of entities that are symmetrically related
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
        return (
            len(self.text_text_pairs)
            + len(self.text_image_pairs)
            + len(self.image_text_pairs)
            + len(self.image_image_pairs)
        )

    def __getitem__(self, idx):
        if idx < len(self.text_text_pairs):
            return {
                "text1": self.text_text_pairs[idx][0],
                "text2": self.text_text_pairs[idx][1],
                "label": self.labels[idx],
            }
        if idx < len(self.text_text_pairs) + len(self.text_image_pairs):
            idx = idx - len(self.text_text_pairs)
            return {
                "text1": self.text_image_pairs[idx][0],
                "image2": self.text_image_pairs[idx][1],
                "label": self.labels[idx],
            }
        if idx < (
            len(self.text_text_pairs)
            + len(self.text_image_pairs)
            + len(self.image_text_pairs)
        ):
            idx = idx - (len(self.text_text_pairs) + len(self.text_image_pairs))
            return {
                "image1": self.image_text_pairs[idx][0],
                "text2": self.image_text_pairs[idx][1],
                "label": self.labels[idx],
            }
        idx = idx - (
            len(self.text_text_pairs)
            + len(self.text_image_pairs)
            + len(self.image_text_pairs)
        )
        return {
            "image1": self.image_image_pairs[idx][0],
            "image2": self.image_image_pairs[idx][1],
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
        model_name = "openai/clip-vit-large-patch14"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the CLIP model and processor
        text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        text_processor = AutoTokenizer.from_pretrained(model_name)

        image_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
        image_processor = AutoProcessor.from_pretrained(model_name).image_processor
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

    def __get_batch_type(self, batch):
        if "text1" in batch and "text2" in batch:
            return "text_text"
        if "text1" in batch and "image2" in batch:
            return "text_image"
        if "image1" in batch and "text2" in batch:
            return "image_text"
        if "image1" in batch and "image2" in batch:
            return "image_image"
        raise ValueError("Invalid batch")

    def fit(self, data_loader: DataLoader):
        criterion = ContrastiveLoss(self.batch_size)
        text_optimizer = optim.AdamW(
            self.text_model.parameters(), lr=self.learning_rate
        )
        image_optimizer = optim.AdamW(
            self.image_model.parameters(), lr=self.learning_rate
        )
        # Fine-tuning loop
        self.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch in data_loader:
                batch_type = self.__get_batch_type(batch)
                if batch_type == "text_text":
                    text_optimizer.zero_grad()

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
                    text_optimizer.step()
                elif batch_type == "text_image":
                    text_optimizer.zero_grad()
                    image_optimizer.zero_grad()

                    text_input = self.text_processor(
                        batch["text1"], return_tensors="pt", padding=True
                    )
                    image_input = self.image_processor(
                        batch["image2"], return_tensors="pt", padding=True
                    )
                    loss = criterion(
                        self.text_model(**text_input).text_embeds,
                        self.image_model(**image_input).image_embeds,
                    )
                    loss.backward()
                    text_optimizer.step()
                    image_optimizer.step()
                elif batch_type == "image_text":
                    text_optimizer.zero_grad()
                    image_optimizer.zero_grad()

                    image_input = self.image_processor(
                        batch["image1"], return_tensors="pt", padding=True
                    )
                    text_input = self.text_processor(
                        batch["text2"], return_tensors="pt", padding=True
                    )
                    loss = criterion(
                        self.image_model(**image_input).image_embeds,
                        self.text_model(**text_input).text_embeds,
                    )
                    loss.backward()
                    text_optimizer.step()
                    image_optimizer.step()
                elif batch_type == "image_image":
                    image_optimizer.zero_grad()

                    image_input_1 = self.image_processor(
                        batch["image1"], return_tensors="pt", padding=True
                    )
                    image_input_2 = self.image_processor(
                        batch["image2"], return_tensors="pt", padding=True
                    )
                    loss = criterion(
                        self.image_model(**image_input_1).image_embeds,
                        self.image_model(**image_input_2).image_embeds,
                    )
                    loss.backward()
                    image_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Loss: {avg_loss}")

    def predict_text(self, text_data):
        self.eval()
        with torch.no_grad():
            # get the text embedding
            text_input = self.text_processor(
                text_data, return_tensors="pt", padding=True
            )
            text_embedding = self.text_model(**text_input).text_embeds
            return text_embedding

    def predict_image(self, image_data):
        self.eval()
        with torch.no_grad():
            # get the image embedding
            image_input = self.image_processor(
                image_data, return_tensors="pt", padding=True
            )
            image_embedding = self.image_model(**image_input).image_embeds
            return image_embedding

    def evaluate(
        self, text_data_1=None, text_data_2=None, image_data_1=None, image_data_2=None
    ):
        if (
            sum(
                [
                    1 if text_data_1 is not None else 0,
                    1 if text_data_2 is not None else 0,
                    1 if image_data_1 is not None else 0,
                    1 if image_data_2 is not None else 0,
                ]
            )
            != 2
        ):
            raise ValueError(
                "Exactly two of text_data_1, text_data_2, image_data_1, image_data_2 must be provided"
            )
        if text_data_1 is not None and text_data_2 is not None:
            data_1 = text_data_1
            data_2 = text_data_2
            kind_1 = "text1"
            kind_2 = "text2"
            predict1 = self.predict_text
            predict2 = self.predict_text
        elif text_data_1 is not None and image_data_2 is not None:
            data_1 = text_data_1
            data_2 = image_data_2
            kind_1 = "text1"
            kind_2 = "image2"
            predict1 = self.predict_text
            predict2 = self.predict_image
        elif image_data_1 is not None and text_data_2 is not None:
            data_1 = image_data_1
            data_2 = text_data_2
            kind_1 = "image1"
            kind_2 = "text2"
            predict1 = self.predict_image
            predict2 = self.predict_text
        elif image_data_1 is not None and image_data_2 is not None:
            data_1 = image_data_1
            data_2 = image_data_2
            kind_1 = "image1"
            kind_2 = "image2"
            predict1 = self.predict_image
            predict2 = self.predict_image
        self.eval()
        with torch.no_grad():
            # calculate the cosine similarity between the two embeddings
            embedding_1 = predict1(data_1)
            embedding_2 = predict2(data_2)
            similarity_scores = nn.CosineSimilarity(dim=1)(embedding_1, embedding_2)
            print(
                pd.DataFrame.from_dict(
                    {
                        kind_1: data_1,
                        kind_2: data_2,
                        "similarity": similarity_scores.tolist(),
                    }
                )
            )


def plot_text_embeddings(model, text_data, save_to: Optional[str] = None):
    tsne = TSNE(n_components=2, random_state=0, metric="cosine", perplexity=5)
    visualisation_set = text_data
    text_embeddings = model.predict_text(visualisation_set).cpu().numpy()
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
        title_x=0.5,
        title_y=0.9,
        title_font_size=30,
    )
    if save_to is not None:
        fig.write_image(save_to)
    fig.show()


def plot_image_embeddings(
    model, image_data: list[tuple[str, np.ndarray]], save_to: Optional[str] = None
):
    tsne = TSNE(
        n_components=2,
        random_state=0,
        metric="cosine",
        perplexity=min(5, len(image_data) - 1),
    )
    names, visualisation_set = list(zip(*image_data))
    names = list(names)
    visualisation_set = list(visualisation_set)
    image_embeddings = model.predict_image(visualisation_set).cpu().numpy()
    image_embeddings = tsne.fit_transform(image_embeddings)
    image_embeddings = pd.DataFrame(
        np.hstack((image_embeddings, np.array(names).reshape(-1, 1))),
        columns=["x", "y", "image"],
    )
    image_embeddings["x"] = image_embeddings["x"].astype(float)
    image_embeddings["y"] = image_embeddings["y"].astype(float)
    fig = px.scatter(image_embeddings, x="x", y="y", text="image")
    fig.update_traces(textposition="top center")
    fig.update_layout(
        height=800,
        title_x=0.5,
        title_y=0.9,
        title_font_size=30,
    )
    if save_to is not None:
        fig.write_image(save_to)
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

    dataset = KGDataset(
        "../results/12_years_a_slave.csv",
        "/Users/nate/Downloads/Faces-2/faces",
        max_pairs=1000,
    )

    image_eval_1 = [
        (name, images[0]) for name, images in dataset.match_face_images(eval_1).items()
    ]
    image_eval_2 = [
        (name, images[0]) for name, images in dataset.match_face_images(eval_2).items()
    ]

    print("Pre-training:")
    model.evaluate(eval_1, eval_2)
    plot_text_embeddings(
        model, list(set(eval_1 + eval_2)), save_to="pre_embeddings_text.svg"
    )
    plot_image_embeddings(
        model,
        list(set(image_eval_1 + image_eval_2)),
        save_to="pre_embeddings_image.svg",
    )
    model.fit(
        DataLoader(
            dataset,
            batch_sampler=KGDatasetBatchSampler(dataset, batch_size),
        )
    )
    print("Post-training:")
    model.evaluate(eval_1, eval_2)
    output_dir = "./fine_tuned_clip_model"
    model.save_pretrained(output_dir)
    print(f"Saved model to '{output_dir}'")
    plot_text_embeddings(
        model, list(set(eval_1 + eval_2)), save_to="post_embeddings_text.svg"
    )
    plot_image_embeddings(
        model,
        list(set(image_eval_1 + image_eval_2)),
        save_to="post_embeddings_image.svg",
    )
