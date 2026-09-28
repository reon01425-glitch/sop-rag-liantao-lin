"""
Multi-view Graph Experts: Entity Graph (G_E), Causal Graph (G_C), and Flow Graph (G_F).
Implements the graph structures, indexing, and scoring functions R_E, R_C, R_F
as formulated in SOPRAG Section 3.2.2 and Section 3.3.3.
"""

import re
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from sop_rag.schema import SOPDocument, Entity, CausalRelation, FlowGraphData, SOPStep
from sop_rag.embeddings import get_embedding_engine


class EntityGraphExpert:
    """
    Entity Graph (G_E):
    Maps discrete domain entities (Alarms, Parameters, Assets, Roles)
    directly to governing Procedure Cards.
    Formula (7):
    R_E = 1 / |V_Q| * sum_{e in V_Q} [ alpha * I(e in V_Entity) + (1 - alpha) * max_{e' in V_Entity} Sim(e, e') ]
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.graph = nx.MultiDiGraph()
        self.sop_entities: Dict[str, List[Entity]] = {}
        self.embedding_engine = get_embedding_engine()

    def add_sop_entities(self, sop_id: str, entities: List[Entity]):
        self.sop_entities[sop_id] = entities
        for ent in entities:
            # Add entity node and edge: (entity) --[ASSOCIATED_WITH]--> (sop_id)
            self.graph.add_node(ent.name, type="entity", category=ent.category, attributes=ent.attributes)
            self.graph.add_node(sop_id, type="procedure_card")
            self.graph.add_edge(ent.name, sop_id, relation="associated_with")

    def score(self, query_entities: List[str], sop_id: str) -> float:
        """Calculate R_E score between query entities and a given SOP."""
        if not query_entities:
            return 0.0

        target_entities = self.sop_entities.get(sop_id, [])
        if not target_entities:
            return 0.0

        target_names = [e.name for e in target_entities]
        target_names_lower = [name.lower() for name in target_names]

        total_score = 0.0
        for q_ent in query_entities:
            q_ent_clean = q_ent.strip().lower()
            if not q_ent_clean:
                continue

            # Exact match check I(e in V_Entity)
            # Match either exact name or substring containment for alarm codes / parameter tags
            exact_match = 0.0
            for t_name_l in target_names_lower:
                if q_ent_clean == t_name_l or (len(q_ent_clean) >= 3 and q_ent_clean in t_name_l) or (len(t_name_l) >= 3 and t_name_l in q_ent_clean):
                    exact_match = 1.0
                    break

            # Semantic similarity max_{e' in V_Entity} Sim(e, e')
            sims = self.embedding_engine.batch_similarity(q_ent, target_names)
            max_sim = max(sims) if sims else 0.0

            # Formula (7)
            term_score = self.alpha * exact_match + (1.0 - self.alpha) * max_sim
            total_score += term_score

        return float(total_score / max(1, len(query_entities)))


class CausalGraphExpert:
    """
    Causal Graph (G_C):
    Models non-linear condition-action logic and diagnostic states:
    G_C = {(s_u -> s_v)} U {(s, PC_i)}
    Formula (8):
    R_C = max_{s in V_State} [ Sim(Q, s) * Path(s -> PC_i) ]
    where Path(s -> PC_i) measures reachability via k-hop traversal.
    """

    def __init__(self, max_hops: int = 4, hop_decay: float = 0.85):
        self.graph = nx.DiGraph()
        self.max_hops = max_hops
        self.hop_decay = hop_decay
        self.all_states: Set[str] = set()
        self.sop_direct_states: Dict[str, Set[str]] = {}
        self.embedding_engine = get_embedding_engine()

    def add_sop_causal(self, sop_id: str, states: List[str], relations: List[CausalRelation]):
        self.graph.add_node(sop_id, type="procedure_card")
        if sop_id not in self.sop_direct_states:
            self.sop_direct_states[sop_id] = set()

        for s in states:
            self.graph.add_node(s, type="state")
            self.all_states.add(s)
            self.sop_direct_states[sop_id].add(s)
            # Direct link from resolving state to SOP
            self.graph.add_edge(s, sop_id, relation="resolves_in")

        for rel in relations:
            self.graph.add_node(rel.cause_state, type="state")
            self.graph.add_node(rel.effect_state, type="state")
            self.all_states.add(rel.cause_state)
            self.all_states.add(rel.effect_state)
            self.graph.add_edge(rel.cause_state, rel.effect_state, relation=rel.relation_type)

    def calculate_path_reachability(self, start_state: str, sop_id: str) -> float:
        """
        Computes Path(s -> PC_i).
        If direct edge: 1.0
        If multi-hop path exists: hop_decay ** (path_len - 1)
        If no path or exceeds max_hops: 0.0
        """
        if not self.graph.has_node(start_state) or not self.graph.has_node(sop_id):
            return 0.0

        if self.graph.has_edge(start_state, sop_id):
            return 1.0

        try:
            path_len = nx.shortest_path_length(self.graph, source=start_state, target=sop_id)
            if path_len <= self.max_hops:
                # hop_decay applied per intermediate transition
                return float(self.hop_decay ** (path_len - 1))
            return 0.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def score(self, query: str, sop_id: str) -> float:
        """
        Calculates R_C = max_{s in V_State} (Sim(Q, s) * Path(s -> PC_i)).
        """
        if not self.all_states or not self.graph.has_node(sop_id):
            return 0.0

        state_list = list(self.all_states)
        sims = self.embedding_engine.batch_similarity(query, state_list)

        max_rc = 0.0
        for state, sim in zip(state_list, sims):
            # Optimization: prune low semantic similarity states (< 0.25)
            if sim < 0.25:
                continue

            reachability = self.calculate_path_reachability(state, sop_id)
            if reachability > 0.0:
                rc_val = sim * reachability
                if rc_val > max_rc:
                    max_rc = rc_val

        return float(max_rc)


class FlowGraphExpert:
    """
    Flow Graph (G_F):
    Local directed flowchart preserving the sequential integrity of execution steps.
    G_F(PC_i) = (V_Step, E_Flow)
    Formula (9):
    R_F = max_{p in V_Step(PC_i)} Sim(Q, p)
    """

    def __init__(self):
        self.flow_graphs: Dict[str, nx.DiGraph] = {}
        self.sop_steps: Dict[str, List[SOPStep]] = {}
        self.embedding_engine = get_embedding_engine()

    def add_sop_flow(self, sop_id: str, flow_data: FlowGraphData):
        g = nx.DiGraph()
        self.sop_steps[sop_id] = flow_data.steps

        for step in flow_data.steps:
            g.add_node(step.step_id, order=step.order, text=step.text, step_type=step.step_type, tool_trigger=step.tool_trigger)

        for src, dst in flow_data.edges:
            g.add_edge(src, dst)

        self.flow_graphs[sop_id] = g

    def score(self, query: str, sop_id: str) -> float:
        """
        Calculates R_F = max_{p in V_Step(PC_i)} Sim(Q, p)
        """
        steps = self.sop_steps.get(sop_id, [])
        if not steps:
            return 0.0

        step_texts = [s.text for s in steps]
        sims = self.embedding_engine.batch_similarity(query, step_texts)
        return float(max(sims)) if sims else 0.0

    def get_linearized_flow(self, sop_id: str) -> List[str]:
        """Returns ordered, verified step sequence for prompt construction."""
        steps = self.sop_steps.get(sop_id, [])
        sorted_steps = sorted(steps, key=lambda x: x.order)
        res = []
        for s in sorted_steps:
            if s.step_type == "condition":
                res.append(f"Step {s.order} [Check]: {s.text}")
            else:
                res.append(f"Step {s.order}: {s.text}")
        return res
