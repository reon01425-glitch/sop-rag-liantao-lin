"""
Unit tests for the SOPRAG framework.
Validates:
- Procedure Card sparse activation
- Entity Graph exact match and semantic scoring
- Causal Graph multi-hop reachability
- Flow Graph topological scoring
- LLM Router weight modulation
- End-to-end pipeline execution and quality scoring
"""

import pytest
from sop_rag.benchmark.dataset import BenchmarkCorpus
from sop_rag.pipeline import SOPRAGPipeline
from sop_rag.router import LLMRouter
from sop_rag.procedure_card import ProcedureCardLayer
from sop_rag.graph_experts import EntityGraphExpert, CausalGraphExpert, FlowGraphExpert
from sop_rag.generator import StructureAwareGenerator
from sop_rag.agent_execution import SpecializedAgentDispatcher


@pytest.fixture
def sample_corpus():
    return BenchmarkCorpus.get_all_sops()


def test_procedure_card_layer(sample_corpus):
    pc_layer = ProcedureCardLayer(top_k=3)
    for s in sample_corpus:
        if s.procedure_card:
            pc_layer.add_card(s.procedure_card)

    results = pc_layer.anchor_search("server outage host reboot", k=3)
    assert len(results) == 3
    top_id, top_score = results[0]
    assert "DC-SOP-CRIT-005.A" in top_id or "DC-SOP-CRIT-005.B" in top_id
    assert top_score > 0.0


def test_entity_graph_expert(sample_corpus):
    entity_expert = EntityGraphExpert(alpha=0.7)
    for s in sample_corpus:
        if s.entities:
            entity_expert.add_sop_entities(s.sop_id, s.entities)

    # Exact match for alarm ALARM-UPS-801
    score_ups = entity_expert.score(["ALARM-UPS-801"], "DC-SOP-UPS-003")
    score_other = entity_expert.score(["ALARM-UPS-801"], "AIR-SOP-UPSET-001")
    assert score_ups > score_other
    assert score_ups > 0.7


def test_causal_graph_expert(sample_corpus):
    causal_expert = CausalGraphExpert(max_hops=4, hop_decay=0.85)
    for s in sample_corpus:
        if s.causal_states or s.causal_relations:
            causal_expert.add_sop_causal(s.sop_id, s.causal_states, s.causal_relations)

    # Path from "Kernel Panic" to DC-SOP-CRIT-005.A
    reach = causal_expert.calculate_path_reachability("Kernel Panic", "DC-SOP-CRIT-005.A")
    assert reach > 0.0

    score_c = causal_expert.score("Why is kernel panic causing server blackout?", "DC-SOP-CRIT-005.A")
    assert score_c > 0.0


def test_flow_graph_expert(sample_corpus):
    flow_expert = FlowGraphExpert()
    for s in sample_corpus:
        if s.flow_graph:
            flow_expert.add_sop_flow(s.sop_id, s.flow_graph)

    # Query matching action step in Upset recovery
    q = "disconnect autopilot and autothrottle immediately"
    score_upset = flow_expert.score(q, "AIR-SOP-UPSET-001")
    score_dc = flow_expert.score(q, "DC-SOP-PUE-012")
    assert score_upset > score_dc


def test_llm_router_gating():
    router = LLMRouter()

    # Case 1: Entity query
    r1 = router.route("What is the status of V-201 and V-202 ball valves in cooling loop?")
    assert r1.w_E > r1.w_C

    # Case 2: Causal query
    r2 = router.route("Why does cold aisle leakage cause PUE spike and thermal overload?")
    assert r2.w_C > 0.3

    # Case 3: Flow query
    r3 = router.route("What is the standard procedure for recovering from an aircraft upset?")
    assert r3.w_F > 0.6


def test_structure_aware_generation(sample_corpus):
    generator = StructureAwareGenerator()
    target_sop = [s for s in sample_corpus if s.sop_id == "DC-SOP-CRIT-005.A"][0]
    out = generator.generate("How to reboot frozen host?", target_sop)

    assert out.sop_quality_score["score"] >= 0.75
    assert out.sop_quality_score["has_step_by_step"] == 1
    assert "Step 1" in out.generated_response
    assert "iDRAC" in out.generated_response or "ILO" in out.generated_response


def test_agent_dispatcher(sample_corpus):
    dispatcher = SpecializedAgentDispatcher()
    sop = [s for s in sample_corpus if s.sop_id == "DC-SOP-CRIT-005.A"][0]
    workflow = dispatcher.plan_workflow(sop)

    assert len(workflow) >= 3
    agent_types = [item["agent_spec"]["agent_type"] for item in workflow]
    assert "Call Agent" in agent_types
    assert "GUI / Terminal Agent" in agent_types


def test_end_to_end_pipeline(sample_corpus):
    pipeline = SOPRAGPipeline()
    pipeline.load_documents(sample_corpus)

    res = pipeline.query("How to initiate a hard server reboot using iDRAC or ILO console?", top_k=3)
    assert len(res["retrieval_results"]) == 3
    assert res["retrieval_results"][0].sop_id == "DC-SOP-CRIT-005.A"
    assert res["generation"] is not None
    assert res["agent_workflow"] is not None
