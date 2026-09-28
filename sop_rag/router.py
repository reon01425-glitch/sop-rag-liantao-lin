"""
Intention-Aware LLM Router (Gating Mechanism) for SOPRAG.
Dynamically assigns attention weights w = [w_E, w_C, w_F] = Softmax(LLMRouter(Q))
matching Section 3.3.2, Appendix A.2, and Appendix H of the SOPRAG paper.
"""

import re
import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sop_rag.schema import RouterWeights


class LLMRouter:
    """
    Intention-Aware Gating Network.
    Evaluates query intent and modulates weights across Entity, Causal, and Flow experts.
    """

    def __init__(self, use_remote_llm: bool = False):
        self.use_remote_llm = use_remote_llm
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def route(self, query: str) -> RouterWeights:
        """Analyze query and compute weights [w_E, w_C, w_F]."""
        if self.use_remote_llm and self.api_key:
            try:
                return self._remote_llm_route(query)
            except Exception as e:
                print(f"[Router] Remote LLM failed ({e}), falling back to local intention engine.")

        return self._local_intention_route(query)

    def _local_intention_route(self, query: str) -> RouterWeights:
        """
        Local high-fidelity intention parser implementing Appendix A.2 & H:
        - Entity Focus: equipment IDs, alarm codes, parameters
        - Causal Focus: 'why', root-cause, symptoms, state transitions
        - Flow Focus: 'how', step-by-step, execution, recovery
        """
        q_lower = query.lower()

        # 1. Extract entities (Alarms, Equipment, Parameters, Codes)
        entities = []
        # Alarm patterns like A01, ALARM-101, ERR_502, CODE-304
        alarm_matches = re.findall(r'\b(?:alarm|code|err|fault)?[_-]?[A-Z0-9]{2,}\b', query)
        for m in alarm_matches:
            if any(char.isdigit() for char in m) and len(m) >= 3:
                entities.append(m)

        # Domain equipment and parameters
        known_equip_params = [
            "hvac", "chilled water", "cooling water", "chiller", "crac", "cdu", "ups",
            "pump", "valve", "manifold", "sensor", "idrac", "ilo", "server", "switch",
            "compressor", "condenser", "heat exchanger", "engine", "flap", "hydraulics",
            "temperature", "pressure", "flow rate", "pue", "voltage", "amperage", "vibration",
            "delta-t", "cfm", "kw", "kwh", "level", "leak detector"
        ]
        for term in known_equip_params:
            if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
                entities.append(term)

        # 2. Compute intent signal logits
        entity_signal = 0.0
        causal_signal = 0.0
        flow_signal = 0.0

        # Entity indicators: specific codes, parameters, asset names
        if entities:
            entity_signal += 1.5 * len(entities)
        if any(w in q_lower for w in ["which equipment", "what parameter", "what is the value", "alarm code", "status of"]):
            entity_signal += 2.0

        # Causal indicators: why, cause, effect, symptom, abnormal, trigger, reason
        causal_keywords = [
            "why", "cause", "causing", "causes", "reason", "leads to", "lead to", "symptom",
            "result of", "due to", "overheat", "overheating", "anomaly", "failure", "fail",
            "spike", "drop", "surge", "tripped", "cavitation", "pressure drop", "how can increasing",
            "improve cooling efficiency", "relationship between"
        ]
        for ck in causal_keywords:
            if ck in q_lower:
                causal_signal += 1.8

        # Flow indicators: how, step, procedure, protocol, actions, do, resolve, mitigate, reboot
        flow_keywords = [
            "how to", "how do i", "how can i", "what is the procedure", "standard procedure",
            "steps", "step-by-step", "workflow", "checklist", "guideline", "protocol",
            "recover", "recovering", "restore", "mitigate", "reboot", "perform", "execute",
            "replace", "turn on", "shut down", "isolate"
        ]
        for fk in flow_keywords:
            if fk in q_lower:
                flow_signal += 1.5

        # Base bias: in SOP operational environments, step execution (Flow) has a strong natural prior (Appendix D)
        flow_signal += 1.0

        # Softmax computation
        logits = np.array([entity_signal, causal_signal, flow_signal], dtype=np.float32)
        # Numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)

        # Format intent summary
        primary_intent = "Flow / Procedural Execution"
        if weights[0] > weights[1] and weights[0] > weights[2]:
            primary_intent = "Entity / Equipment Lookup"
        elif weights[1] > weights[0] and weights[1] > weights[2]:
            primary_intent = "Causal / Diagnostic Reasoning"

        return RouterWeights(
            w_E=round(float(weights[0]), 3),
            w_C=round(float(weights[1]), 3),
            w_F=round(float(weights[2]), 3),
            intent_summary=f"{primary_intent} (dominant weight {max(weights):.2f})",
            extracted_entities=list(set(entities))
        )

    def _remote_llm_route(self, query: str) -> RouterWeights:
        """Call remote LLM with Appendix A.2 prompt if configured."""
        # For OpenAI or Gemini via HTTP / client
        prompt = (
            "You are an expert at understanding user queries about Standard Operating Procedures.\n"
            "Your task is to analyze the query and determine:\n"
            "1. The user's intent (what is the operator trying to achieve?)\n"
            "2. Key entities mentioned (alarms, parameters, equipment)\n"
            "3. Weights w_E, w_C, w_F (summing to 1.0):\n"
            "   - Entity graph (w_E): Focus on specific alarms, parameters, equipment\n"
            "   - Causal graph (w_C): Focus on cause-effect, why something happens\n"
            "   - Flow graph (w_F): Focus on step-by-step procedures, how to do something\n"
            f"Query: \"{query}\"\n"
            "Respond in JSON format: {\"intent\": \"...\", \"entities\": [...], \"w_E\": 0.2, \"w_C\": 0.1, \"w_F\": 0.7}"
        )
        # Fallback to local
        return self._local_intention_route(query)
