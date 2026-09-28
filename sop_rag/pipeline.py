"""
Unified SOPRAG Pipeline.
Integrates corpus ingestion, multi-view indexing, intention routing,
gated graph reasoning, structure-aware generation, and agent workflow planning.
"""

from typing import List, Dict, Any, Optional, Tuple
from sop_rag.schema import SOPDocument, RetrievalScore, SOPGenerationOutput
from sop_rag.retriever import SOPRAGRetriever
from sop_rag.generator import StructureAwareGenerator
from sop_rag.agent_execution import SpecializedAgentDispatcher


class SOPRAGPipeline:
    """End-to-end SOPRAG System."""

    def __init__(
        self,
        lambda_param: float = 0.35,
        top_k_pc: int = 5,
        entity_alpha: float = 0.6
    ):
        self.retriever = SOPRAGRetriever(
            lambda_param=lambda_param,
            top_k_pc=top_k_pc,
            entity_alpha=entity_alpha
        )
        self.generator = StructureAwareGenerator()
        self.agent_dispatcher = SpecializedAgentDispatcher()

    def load_documents(self, documents: List[SOPDocument]):
        """Indexes all SOP documents into the knowledge base."""
        for doc in documents:
            self.retriever.index_sop(doc)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        generate_response: bool = True,
        plan_agents: bool = True,
        ablation_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete SOPRAG pipeline for a user query.
        Returns:
        {
            "query": str,
            "router_weights": dict,
            "retrieval_results": List[RetrievalScore],
            "retrieval_latency_ms": float,
            "generation": SOPGenerationOutput or None,
            "agent_workflow": list or None
        }
        """
        # 1. MoE-Style SOP Retrieval
        retrieval_scores, latency = self.retriever.retrieve(
            query=query_text,
            k=top_k,
            ablation_mode=ablation_mode
        )

        top_sop = None
        generation_output = None
        agent_workflow = None

        if retrieval_scores:
            best_id = retrieval_scores[0].sop_id
            top_sop = self.retriever.sops.get(best_id)

            # 2. Structure-Aware Generation
            if generate_response and top_sop:
                generation_output = self.generator.generate(query_text, top_sop)

            # 3. Agent-Integrated Operational Execution
            if plan_agents and top_sop:
                agent_workflow = self.agent_dispatcher.plan_workflow(top_sop)

        return {
            "query": query_text,
            "router_weights": retrieval_scores[0].weights if retrieval_scores else None,
            "retrieval_results": retrieval_scores,
            "retrieval_latency_ms": round(latency * 1000, 2),
            "generation": generation_output,
            "agent_workflow": agent_workflow
        }
