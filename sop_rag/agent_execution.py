"""
Agent-Integrated Operational Execution (Appendix G, Figure 7).
Transforms procedural steps within Flow Graph G_F into executable agentic workflows
with human-in-the-loop confirmation.
"""

from typing import List, Dict, Any, Optional
from sop_rag.schema import SOPDocument, SOPStep


class SpecializedAgentDispatcher:
    """Dispatches flow graph steps to specialized industrial operational agents."""

    @staticmethod
    def identify_agent_action(step: SOPStep) -> Dict[str, Any]:
        """Detects if step triggers Call Agent, Email Agent, Terminal/GUI Agent, or manual action."""
        text_lower = step.text.lower()

        # Call Agent trigger
        if any(w in text_lower for w in ["call", "dial", "phone", "notify hotline", "contact supervisor", "ring"]):
            return {
                "agent_type": "Call Agent",
                "tool_name": "telecom_dialer",
                "action": "Prepare voice dispatch",
                "target": "On-Duty Shift Supervisor / Incident Commander",
                "requires_confirmation": True,
                "preview": f"Dial emergency line for Step {step.order}: {step.text[:60]}..."
            }

        # Email / Notification Agent trigger
        if any(w in text_lower for w in ["email", "notify team", "log incident", "broadcast", "slack", "dispatch notice"]):
            return {
                "agent_type": "Email Agent",
                "tool_name": "incident_mailer",
                "action": "Draft notification message",
                "target": "Ops-Distribution-List <ops-team@industrial-facility.internal>",
                "requires_confirmation": True,
                "preview": f"Send incident alert for Step {step.order}: {step.text[:60]}..."
            }

        # DCIM / SCADA / Terminal GUI Agent trigger
        if any(w in text_lower for w in ["reboot", "idrac", "ilo", "scada", "dcim", "valve", "pump", "isolate", "reset", "switch"]):
            return {
                "agent_type": "GUI / Terminal Agent",
                "tool_name": "dcim_scada_controller",
                "action": "Stage hardware / telemetry execution",
                "target": "DCIM / BMS Central Control Gateway",
                "requires_confirmation": True,
                "preview": f"Stage control signal for Step {step.order}: {step.text[:60]}..."
            }

        return {
            "agent_type": "Operator Manual Action",
            "tool_name": "manual_inspection",
            "action": "Manual visual check or tool handling",
            "target": "Frontline Field Technician",
            "requires_confirmation": False,
            "preview": f"Manual step: {step.text}"
        }

    def plan_workflow(self, sop: SOPDocument) -> List[Dict[str, Any]]:
        """Plans complete multi-agent operational workflow for a retrieved SOP."""
        if not sop.flow_graph:
            return []

        workflow = []
        for step in sorted(sop.flow_graph.steps, key=lambda s: s.order):
            agent_spec = self.identify_agent_action(step)
            workflow.append({
                "step_order": step.order,
                "step_text": step.text,
                "step_type": step.step_type,
                "agent_spec": agent_spec
            })
        return workflow
