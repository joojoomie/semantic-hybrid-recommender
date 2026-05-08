from __future__ import annotations

import torch
from torch import nn


class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 32) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        return torch.einsum("bd,bd->b", user_vec, item_vec)


class PairwiseInteraction(nn.Module):
    def forward(self, feature_vectors: torch.Tensor) -> torch.Tensor:
        pairwise_dots = []
        num_features = feature_vectors.size(1)
        for left in range(num_features):
            for right in range(left + 1, num_features):
                pairwise_dots.append(
                    torch.einsum(
                        "bd,bd->b",
                        feature_vectors[:, left, :],
                        feature_vectors[:, right, :],
                    )
                )
        return torch.stack(pairwise_dots, dim=1)


class MiniDLRM(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        dense_dim: int,
        emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.category_embedding = nn.Embedding(num_categories, emb_dim)
        self.dense_bottom = nn.Sequential(
            nn.Linear(dense_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        self.interaction = PairwiseInteraction()
        num_features = 4
        num_pairs = num_features * (num_features - 1) // 2
        top_input_dim = emb_dim + num_pairs
        self.top_mlp = nn.Sequential(
            nn.Linear(top_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in (self.user_embedding, self.item_embedding, self.category_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        category_vec = self.category_embedding(category_idx)
        dense_vec = self.dense_bottom(dense_features)
        feature_vectors = torch.stack([user_vec, item_vec, category_vec, dense_vec], dim=1)
        interactions = self.interaction(feature_vectors)
        top_input = torch.cat([dense_vec, interactions], dim=1)
        return self.top_mlp(top_input).squeeze(1)


class HybridDLRM(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        dense_dim: int,
        semantic_embeddings,
        semantic_dim: int,
        emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        self.category_embedding = nn.Embedding(num_categories, emb_dim)
        self.dense_bottom = nn.Sequential(
            nn.Linear(dense_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
        )
        semantic_tensor = torch.as_tensor(semantic_embeddings, dtype=torch.float32)
        if semantic_tensor.shape != (num_items, semantic_dim):
            raise ValueError(
                f"semantic_embeddings must have shape {(num_items, semantic_dim)}, "
                f"got {tuple(semantic_tensor.shape)}"
            )
        self.semantic_embedding_table = nn.Embedding.from_pretrained(
            semantic_tensor,
            freeze=True,
        )
        self.semantic_projection = nn.Linear(semantic_dim, emb_dim)
        self.interaction = PairwiseInteraction()
        num_features = 5
        num_pairs = num_features * (num_features - 1) // 2
        top_input_dim = emb_dim + num_pairs
        self.top_mlp = nn.Sequential(
            nn.Linear(top_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in (self.user_embedding, self.item_embedding, self.category_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        category_idx: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.user_embedding(user_idx)
        item_vec = self.item_embedding(item_idx)
        category_vec = self.category_embedding(category_idx)
        dense_vec = self.dense_bottom(dense_features)
        semantic_vec = self.semantic_embedding_table(item_idx)
        semantic_proj = self.semantic_projection(semantic_vec)
        feature_vectors = torch.stack(
            [user_vec, item_vec, category_vec, dense_vec, semantic_proj],
            dim=1,
        )
        interactions = self.interaction(feature_vectors)
        top_input = torch.cat([dense_vec, interactions], dim=1)
        return self.top_mlp(top_input).squeeze(1)
