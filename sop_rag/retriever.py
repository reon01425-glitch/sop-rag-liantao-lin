"""
SOPRAG Retriever & Baseline Retrievers.
Implements:
- SOPRAG MoE Retrieval Pipeline (PC Anchoring + LLM Router + Multi-view Graph Reasoning)
- Baselines: BM25, Dense Retrieval, Co-occurrence GraphRAG
- Ablation models: w/o PC, w/o Entity, w/o Causal, w/o Flow, Full (Uniform Avg)
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi

from sop_rag.schema import SOPDocument, RetrievalScore, RouterWeights
from sop_rag.procedure_card import ProcedureCardLayer
from sop_rag.graph_experts import EntityGraphExpert, CausalGraphExpert, FlowGraphExpert
from sop_rag.router import LLMRouter
from sop_rag.embeddings import get_embedding_engine


class SOPRAGRetriever:
    """
    SOPRAG Retrieval Engine.
    Executes Coarse-to-Fine MoE-gated retrieval over industrial SOPs.
    """

    def __init__(
        self,
        lambda_param: float = 0.35,  # Weight for anchor vs expert subgraphs
        top_k_pc: int = 5,           # Procedure Card prune threshold
        entity_alpha: float = 0.6,
        causal_max_hops: int = 4,
        causal_hop_decay: float = 0.85
    ):
        self.lambda_param = lambda_param
        self.top_k_pc = top_k_pc

        self.pc_layer = ProcedureCardLayer(top_k=top_k_pc)
        self.entity_expert = EntityGraphExpert(alpha=entity_alpha)
        self.causal_expert = CausalGraphExpert(max_hops=causal_max_hops, hop_decay=causal_hop_decay)
        self.flow_expert = FlowGraphExpert()
        self.router = LLMRouter()
        self.embedding_engine = get_embedding_engine()

        self.sops: Dict[str, SOPDocument] = {}

    def index_sop(self, sop: SOPDocument):
        """Indexes an atomic SOP across the Procedure Card layer and 3 graph experts."""
        self.sops[sop.sop_id] = sop

        # 1. Procedure Card Layer
        if sop.procedure_card:
            self.pc_layer.add_card(sop.procedure_card)

        # 2. Entity Graph Expert
        if sop.entities:
            self.entity_expert.add_sop_entities(sop.sop_id, sop.entities)

        # 3. Causal Graph Expert
        if sop.causal_states or sop.causal_relations:
            self.causal_expert.add_sop_causal(sop.sop_id, sop.causal_states, sop.causal_relations)

        # 4. Flow Graph Expert
        if sop.flow_graph:
            self.flow_expert.add_sop_flow(sop.sop_id, sop.flow_graph)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        ablation_mode: Optional[str] = None
    ) -> Tuple[List[RetrievalScore], float]:
        """
        Retrieves top-k relevant SOPs for a given query Q.
        ablation_mode options:
          - None: Standard full SOPRAG
          - "wo_pc": No Procedure Card pruning (search all SOPs)
          - "wo_entity": w_E = 0
          - "wo_causal": w_C = 0
          - "wo_flow": w_F = 0
          - "full_avg": w_E = w_C = w_F = 1/3 (static routing)
        Returns: (ranked_scores, latency_in_seconds)
        """
        start_time = time.perf_counter()

        # Step 1: Procedure Card Layer Anchoring (Section 3.3.1)
        if ablation_mode == "wo_pc":
            # Consider all SOPs without pruning
            candidate_cards = [(sid, self.embedding_engine.similarity(query, f"{self.sops[sid].title} - {self.sops[sid].procedure_card.abstract if self.sops[sid].procedure_card else ''}")) for sid in self.sops]
        else:
            # Sparse activation: retrieve Top-K Procedure Cards
            candidate_cards = self.pc_layer.anchor_search(query, k=self.top_k_pc)

        # Step 2: LLM-Guided Routing (Section 3.3.2)
        router_weights = self.router.route(query)
        w_E, w_C, w_F = router_weights.w_E, router_weights.w_C, router_weights.w_F

        # Apply ablation modifications if requested
        if ablation_mode == "wo_entity":
            w_E = 0.0
            sum_w = w_C + w_F
            if sum_w > 0:
                w_C, w_F = w_C / sum_w, w_F / sum_w
        elif ablation_mode == "wo_causal":
            w_C = 0.0
            sum_w = w_E + w_F
            if sum_w > 0:
                w_E, w_F = w_E / sum_w, w_F / sum_w
        elif ablation_mode == "wo_flow":
            w_F = 0.0
            sum_w = w_E + w_C
            if sum_w > 0:
                w_E, w_C = w_E / sum_w, w_C / sum_w
        elif ablation_mode == "full_avg":
            w_E, w_C, w_F = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

        effective_weights = RouterWeights(
            w_E=round(w_E, 3),
            w_C=round(w_C, 3),
            w_F=round(w_F, 3),
            intent_summary=router_weights.intent_summary,
            extracted_entities=router_weights.extracted_entities
        )

        # Step 3: Subgraphs Reasoning & Aggregated Scoring (Section 3.3.3)
        scores: List[RetrievalScore] = []
        for sop_id, anchor_sim in candidate_cards:
            sop = self.sops[sop_id]

            # 1. Entity Score R_E
            r_E = 0.0
            if w_E > 0.0:
                r_E = self.entity_expert.score(router_weights.extracted_entities, sop_id)

            # 2. Causal Score R_C
            r_C = 0.0
            if w_C > 0.0:
                r_C = self.causal_expert.score(query, sop_id)

            # 3. Flow Score R_F
            r_F = 0.0
            if w_F > 0.0:
                r_F = self.flow_expert.score(query, sop_id)

            # Aggregation: Formula (6)
            # R(Q, S_i) = lambda * Sim(Q, PC_i) + (1 - lambda) * sum_m (w_m * R_m)
            expert_weighted_sum = (w_E * r_E) + (w_C * r_C) + (w_F * r_F)
            final_score = (self.lambda_param * anchor_sim) + ((1.0 - self.lambda_param) * expert_weighted_sum)

            scores.append(
                RetrievalScore(
                    sop_id=sop_id,
                    title=sop.title,
                    final_score=round(float(final_score), 4),
                    anchor_score=round(float(anchor_sim), 4),
                    entity_score=round(float(r_E), 4),
                    causal_score=round(float(r_C), 4),
                    flow_score=round(float(r_F), 4),
                    weights=effective_weights
                )
            )

        # Sort descending by final relevance score
        scores.sort(key=lambda x: x.final_score, reverse=True)
        latency = time.perf_counter() - start_time
        return scores[:k], latency


class BaselineBM25Retriever:
    """Standard probabilistic lexical retrieval based on exact term matching."""

    def __init__(self, sops: Dict[str, SOPDocument]):
        self.sops = sops
        self.sop_ids = list(sops.keys())
        self.corpus = [sops[sid].raw_markdown.lower().split() for sid in self.sop_ids]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, float]], float]:
        start = time.perf_counter()
        tokenized_q = query.lower().split()
        scores = self.bm25.get_scores(tokenized_q)
        ranked = sorted(zip(self.sop_ids, scores), key=lambda x: x[1], reverse=True)
        latency = time.perf_counter() - start
        return ranked[:k], latency


class BaselineDenseRetriever:
    """Dense semantic retrieval using text embeddings (e.g. OpenAI Embed / BGE-small)."""

    def __init__(self, sops: Dict[str, SOPDocument]):
        self.sops = sops
        self.sop_ids = list(sops.keys())
        self.embedding_engine = get_embedding_engine()
        self.doc_texts = [sops[sid].raw_markdown for sid in self.sop_ids]
        self.doc_embeddings = self.embedding_engine.embed_texts(self.doc_texts)

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, float]], float]:
        start = time.perf_counter()
        q_emb = self.embedding_engine.embed_text(query)
        sims = np.dot(self.doc_embeddings, q_emb)
        ranked = sorted(zip(self.sop_ids, sims.tolist()), key=lambda x: x[1], reverse=True)
        latency = time.perf_counter() - start
        return ranked[:k], latency


class BaselineGraphRAGRetriever:
    """
    Standard GraphRAG baseline:
    Builds a flat text co-occurrence graph across chunks, aggregating
    semantic similarity without procedural hierarchy or MoE routing.
    """

    def __init__(self, sops: Dict[str, SOPDocument]):
        self.sops = sops
        self.sop_ids = list(sops.keys())
        self.embedding_engine = get_embedding_engine()
        # Chunks extracted across all SOPs
        self.chunks: List[Tuple[str, str]] = []  # (sop_id, chunk_text)
        for sid, doc in sops.items():
            paragraphs = [p.strip() for p in doc.raw_markdown.split("\n\n") if len(p.strip()) > 30]
            for p in paragraphs:
                self.chunks.append((sid, p))

        self.chunk_texts = [c[1] for c in self.chunks]
        self.chunk_embeddings = self.embedding_engine.embed_texts(self.chunk_texts) if self.chunk_texts else np.zeros((0, 384))

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[Tuple[str, float]], float]:
        start = time.perf_counter()
        if len(self.chunks) == 0:
            return [], 0.0

        q_emb = self.embedding_engine.embed_text(query)
        sims = np.dot(self.chunk_embeddings, q_emb)

        # Community-level / chunk aggregation per SOP
        sop_scores: Dict[str, List[float]] = {sid: [] for sid in self.sop_ids}
        for (sid, _), sim in zip(self.chunks, sims):
            sop_scores[sid].append(sim)

        # Average of top-3 chunks per SOP
        agg_scores = []
        for sid, score_list in sop_scores.items():
            if score_list:
                top_chunks = sorted(score_list, reverse=True)[:3]
                agg_scores.append((sid, float(np.mean(top_chunks))))
            else:
                agg_scores.append((sid, 0.0))

        ranked = sorted(agg_scores, key=lambda x: x[1], reverse=True)
        latency = time.perf_counter() - start
        return ranked[:k], latency
