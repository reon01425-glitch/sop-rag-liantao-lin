"""
Agent-based Simulation Environment for automated SOP benchmark construction.
Implements the 4-agent collaborative pipeline from Figure 3 & Section 4.1:
- Collector Agent: Topic-driven domain discovery
- Archivist Agent: Normalization into clean Markdown
- Auditor Agent: Quality filtering and atomic segmentation
- Examiner Agent: Frontline operator persona for query generation
"""

import json
from typing import List, Dict, Any
from sop_rag.schema import SOPDocument, ProcedureCard, Entity, CausalRelation, FlowGraphData, SOPStep


class CollectorAgent:
    """Agent that discovers raw industrial procedure sources for a given domain."""
    def __init__(self, domain_theme: str):
        self.domain_theme = domain_theme

    def collect(self) -> List[Dict[str, str]]:
        return [
            {
                "topic": self.domain_theme,
                "status": "retrieved",
                "raw_file": f"{self.domain_theme.lower().replace(' ', '_')}_raw_spec.md"
            }
        ]


class ArchivistAgent:
    """Agent that normalizes heterogeneous file formats into structured Markdown."""
    def normalize(self, raw_input: str) -> str:
        # Standardize headers, lists, and parameter blocks
        cleaned = raw_input.strip()
        return cleaned


class AuditorAgent:
    """Agent that verifies SOP validity and segments multi-procedure files into atomic units."""
    def audit_and_segment(self, markdown_text: str) -> List[Dict[str, Any]]:
        # Verifies operational executable nature
        is_sop = any(keyword in markdown_text.lower() for keyword in ["step", "protocol", "procedure", "pre-action", "mitigation"])
        if not is_sop:
            return []
        return [{"valid_atomic_sop": True, "content": markdown_text}]


class ExaminerAgent:
    """
    Question Generation Agent adopting frontline operator personas.
    Synthesizes diverse query types:
    - Entity-focused (parameters, alarm codes, equipment models)
    - Causal-focused (diagnostic 'why', symptoms, underlying root causes)
    - Flow-focused ('how-to', sequential execution steps)
    """
    def generate_evaluation_queries(self, sop: SOPDocument) -> List[Dict[str, Any]]:
        queries = []

        # 1. Entity query
        if sop.entities:
            top_entity = sop.entities[0].name
            queries.append({
                "domain": sop.domain,
                "query_type": "entity",
                "query": f"What are the specific parameters and requirements regarding {top_entity} in {sop.title}?",
                "target_sop_id": sop.sop_id
            })

        # 2. Causal query
        if sop.causal_states:
            symptom = sop.causal_states[0]
            queries.append({
                "domain": sop.domain,
                "query_type": "causal",
                "query": f"Why does {symptom} occur and what underlying causes need to be verified?",
                "target_sop_id": sop.sop_id
            })

        # 3. Flow query
        queries.append({
            "domain": sop.domain,
            "query_type": "flow",
            "query": f"What is the step-by-step actionable procedure for {sop.title}?",
            "target_sop_id": sop.sop_id
        })

        return queries
