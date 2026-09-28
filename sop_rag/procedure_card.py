"""
Procedure Card (PC) Layer implementation.
Represents V_Global and performs Sparse Activation Layer Anchoring:
A_TopK = TopK_{PC_i in V_Global}(Sim(Q, PC_i))
Prunes the reasoning space to eliminate computational noise before drilling
into multi-view graph reasoning.
"""

from typing import List, Dict, Tuple
from sop_rag.schema import ProcedureCard
from sop_rag.embeddings import get_embedding_engine


class ProcedureCardLayer:
    """Global Procedure Card Layer (V_Global)."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.cards: Dict[str, ProcedureCard] = {}
        self.card_texts: List[str] = []
        self.sop_ids: List[str] = []
        self.embedding_engine = get_embedding_engine()

    def add_card(self, card: ProcedureCard):
        self.cards[card.sop_id] = card
        self._rebuild_index()

    def add_cards(self, cards: List[ProcedureCard]):
        for c in cards:
            self.cards[c.sop_id] = c
        self._rebuild_index()

    def _rebuild_index(self):
        self.sop_ids = list(self.cards.keys())
        self.card_texts = [f"{self.cards[sid].title} - {self.cards[sid].abstract}" for sid in self.sop_ids]

    def anchor_search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Calculates Sim(Q, PC_i) for all cards in V_Global and returns top-k:
        A_TopK = TopK_{PC_i in V_Global}(Sim(Q, PC_i))
        Returns list of tuples: (sop_id, anchor_similarity_score).
        """
        if not self.sop_ids:
            return []

        limit = k if k is not None else self.top_k
        limit = min(limit, len(self.sop_ids))

        sims = self.embedding_engine.batch_similarity(query, self.card_texts)
        scored_pairs = list(zip(self.sop_ids, sims))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:limit]

    def get_card(self, sop_id: str) -> ProcedureCard:
        return self.cards.get(sop_id)
