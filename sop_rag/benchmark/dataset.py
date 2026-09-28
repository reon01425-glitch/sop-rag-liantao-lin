"""
Rich Industrial Benchmark Dataset for SOPRAG.
Implements 24 detailed atomic SOPs with confusable distractor pairs across 4 domains:
1. Data Center
2. Liquid Cooling
3. Building Management
4. Airline Services

Each SOP includes:
- Procedure Card (Title + Abstract)
- Entity Graph nodes (Alarms, Parameters, Assets, Roles)
- Causal Graph states and transitions (Causes, Mitigates, Leads To)
- Flow Graph steps (Topological order, conditions, actions, agent triggers)
- Evaluation queries spanning Entity, Causal, and Flow intents.
"""

from typing import List, Dict, Tuple, Any
from sop_rag.schema import (
    SOPDocument, ProcedureCard, Entity, CausalRelation, FlowGraphData, SOPStep
)


class BenchmarkCorpus:
    """Comprehensive industrial SOP corpus with confusable pairs."""

    @staticmethod
    def get_data_center_sops() -> List[SOPDocument]:
        return [
            SOPDocument(
                sop_id="DC-SOP-CRIT-005.A",
                domain="data_center",
                document_id="IT-SOP-CRIT-005.A",
                version="4.2",
                title="Critical Bare-Metal Server Outage and Host Reboot Protocol",
                scope="Tier 1 Support and Mission-Critical Compute Nodes under unrecoverable OS freeze",
                pre_action_requirements=["SSH client", "iDRAC/ILO access", "Incident Log Form ILF-01", "Supervisor Contact List"],
                raw_markdown="""# Critical Bare-Metal Server Outage and Host Reboot Protocol
## 1. Scope & Definitions
Applies to Tier 1 Support / Critical Outages Only. Handles unrecoverable bare-metal node freezes.
## 2. Pre-Action Requirements
Required: SSH Client, Supervisor Contact List, Incident Log Form (ILF-01), iDRAC/ILO admin privileges.
## 3. Detailed Mitigation Protocol
- Step 1: Call Supervisor (Incident Commander) immediately upon confirmed node blackout.
- Step 2: Attempt a hard server reboot via iDRAC/ILO out-of-band management console.
- Step 3: Verify system status post-reboot using the Post-Boot Telemetry Verification checklist.
- Step 4: Log incident and notify team (DL-IT-03) with timestamped diagnostic dump.
## 4. Review & Sign-off
Prepared by IT Reliability Lead. Approved by Infrastructure Director.
""",
                procedure_card=ProcedureCard(
                    sop_id="DC-SOP-CRIT-005.A",
                    domain="data_center",
                    title="Critical Bare-Metal Server Outage and Host Reboot Protocol",
                    abstract="Immediate mitigation workflow for host blackouts via out-of-band management reboot and post-boot verification."
                ),
                entities=[
                    Entity(name="iDRAC", category="equipment", attributes={"type": "out-of-band management"}),
                    Entity(name="ILO", category="equipment", attributes={"type": "out-of-band management"}),
                    Entity(name="ILF-01", category="alarm", attributes={"type": "incident form"}),
                    Entity(name="DL-IT-03", category="role", attributes={"type": "notification list"}),
                    Entity(name="Supervisor", category="role", attributes={"role": "Incident Commander"}),
                    Entity(name="Server Reboot", category="parameter", attributes={"action": "power cycle"})
                ],
                causal_states=[
                    "Server Unresponsive", "Kernel Panic", "OS Freeze", "Host Blackout", "Node Outage"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Kernel Panic", effect_state="Server Unresponsive", relation_type="causes"),
                    CausalRelation(cause_state="OS Freeze", effect_state="Host Blackout", relation_type="causes"),
                    CausalRelation(cause_state="Server Unresponsive", effect_state="Node Outage", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="DC-SOP-CRIT-005.A",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Call Supervisor (Incident Commander) immediately upon confirmed node blackout.", step_type="action", tool_trigger="call_agent"),
                        SOPStep(step_id="s2", order=2, text="Attempt a hard server reboot via iDRAC/ILO out-of-band management console.", step_type="action", tool_trigger="dcim_scada_controller"),
                        SOPStep(step_id="s3", order=3, text="Verify system status post-reboot using Post-Boot Telemetry Verification checklist.", step_type="condition"),
                        SOPStep(step_id="s4", order=4, text="Log incident and notify team (DL-IT-03) with timestamped diagnostic dump.", step_type="action", tool_trigger="incident_mailer")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4")]
                )
            ),
            SOPDocument(
                sop_id="DC-SOP-CRIT-005.B",
                domain="data_center",
                document_id="IT-SOP-CRIT-005.B",
                version="3.1",
                title="Virtualized Cluster Hypervisor Crash and Live Migration Failover",
                scope="Virtual machine cluster compute nodes, VMware ESXi / KVM hypervisor isolation",
                pre_action_requirements=["vCenter / OpenStack admin token", "Storage SAN interconnect check", "DR plan DR-99"],
                raw_markdown="""# Virtualized Cluster Hypervisor Crash and Live Migration Failover
## 1. Scope
Triggered when an entire hypervisor node drops out of the virtual cluster quorum without kernel reboot.
## 2. Action Sequence
- Step 1: Query cluster quorum witness via CLI to confirm node isolation.
- Step 2: Trigger automated evacuate / live migration of remaining healthy guest virtual machines to standby compute hosts.
- Step 3: Put faulted hypervisor host into Maintenance Mode and disable HA restart loops.
- Step 4: Collect vmkernel core dump and submit crash ticket to Cloud Platform Engineering.
""",
                procedure_card=ProcedureCard(
                    sop_id="DC-SOP-CRIT-005.B",
                    domain="data_center",
                    title="Virtualized Cluster Hypervisor Crash and Live Migration Failover",
                    abstract="Failover and virtual machine evacuation protocol when hypervisor quorum is lost."
                ),
                entities=[
                    Entity(name="Hypervisor", category="equipment", attributes={"type": "ESXi/KVM"}),
                    Entity(name="Quorum Witness", category="equipment", attributes={"type": "Cluster Master"}),
                    Entity(name="Live Migration", category="parameter", attributes={"type": "vMotion"}),
                    Entity(name="DR-99", category="alarm", attributes={"type": "Disaster Plan"}),
                    Entity(name="Cloud Platform Engineering", category="role", attributes={"team": "Tier 3"})
                ],
                causal_states=[
                    "Hypervisor Crash", "Quorum Split-Brain", "VM Guest Freeze", "Management Agent Timeout"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Management Agent Timeout", effect_state="Hypervisor Crash", relation_type="causes"),
                    CausalRelation(cause_state="Hypervisor Crash", effect_state="Quorum Split-Brain", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="DC-SOP-CRIT-005.B",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Query cluster quorum witness via CLI to confirm node isolation.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Trigger automated evacuate / live migration of remaining healthy guest VMs to standby hosts.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Put faulted hypervisor host into Maintenance Mode and disable HA restart loops.", step_type="condition"),
                        SOPStep(step_id="s4", order=4, text="Collect vmkernel core dump and submit crash ticket to Cloud Platform Engineering.", step_type="action", tool_trigger="incident_mailer")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4")]
                )
            ),
            SOPDocument(
                sop_id="DC-SOP-PUE-012",
                domain="data_center",
                document_id="DC-SOP-PUE-012",
                version="2.1",
                title="PUE Threshold Exceedance and CRAC Thermal Anomaly Response",
                scope="Data Hall Row Cooling, CRAC unit thermal balance, PUE > 1.45 containment",
                pre_action_requirements=["BMS telemetry terminal", "Infrared thermal gun", "CRAC technician credential"],
                raw_markdown="""# PUE Threshold Exceedance and CRAC Thermal Anomaly Response
## 1. Scope
Triggered when Power Usage Effectiveness (PUE) exceeds 1.45 or CRAC unit Delta-T drops below 10°F.
## 2. Mitigation Steps
- Step 1: Check CRAC-3 and CRAC-4 chilled water supply valves for valve actuator sticking.
- Step 2: Inspect cold-aisle containment panels for physical air leakage or door seals open.
- Step 3: Check variable speed fan drive (VFD) RPM settings on air handlers.
- Step 4: Adjust chiller setpoint downwards by 1.5°F if return water temperature exceeds 65°F.
- Step 5: Notify Energy Management team via automated BMS incident report.
""",
                procedure_card=ProcedureCard(
                    sop_id="DC-SOP-PUE-012",
                    domain="data_center",
                    title="PUE Threshold Exceedance and CRAC Thermal Anomaly Response",
                    abstract="Response protocol when PUE exceeds 1.45 or CRAC air handling exhibits thermal imbalance."
                ),
                entities=[
                    Entity(name="PUE", category="parameter", attributes={"threshold": "1.45"}),
                    Entity(name="CRAC-3", category="equipment", attributes={"type": "Computer Room Air Handler"}),
                    Entity(name="CRAC-4", category="equipment", attributes={"type": "Computer Room Air Handler"}),
                    Entity(name="VFD", category="equipment", attributes={"type": "Variable Frequency Drive"}),
                    Entity(name="Delta-T", category="parameter", attributes={"nominal": ">10F"}),
                    Entity(name="Energy Management", category="role", attributes={"team": "Facilities"})
                ],
                causal_states=[
                    "PUE Spike", "High Thermal Load", "Cold Aisle Leakage", "CRAC Valve Sticking", "Return Water High Temp"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Cold Aisle Leakage", effect_state="High Thermal Load", relation_type="causes"),
                    CausalRelation(cause_state="CRAC Valve Sticking", effect_state="Return Water High Temp", relation_type="causes"),
                    CausalRelation(cause_state="High Thermal Load", effect_state="PUE Spike", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="DC-SOP-PUE-012",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Check CRAC-3 and CRAC-4 chilled water supply valves for valve actuator sticking.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Inspect cold-aisle containment panels for physical air leakage or door seals open.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Check variable speed fan drive (VFD) RPM settings on air handlers.", step_type="condition"),
                        SOPStep(step_id="s4", order=4, text="Adjust chiller setpoint downwards by 1.5°F if return water temperature exceeds 65°F.", step_type="action"),
                        SOPStep(step_id="s5", order=5, text="Notify Energy Management team via automated BMS incident report.", step_type="action", tool_trigger="incident_mailer")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="DC-SOP-UPS-003",
                domain="data_center",
                document_id="DC-SOP-UPS-003",
                version="3.0",
                title="Static Transfer Switch and UPS Battery String Alarm Isolation",
                scope="Uninterruptible Power Supply (UPS) Modules A and B, Battery Strings 1-4, STS Transfer",
                pre_action_requirements=["NFPA 70E Arc Flash PPE Category 2", "Fluke Clamp Multimeter", "EPO bypass key"],
                raw_markdown="""# Static Transfer Switch and UPS Battery String Alarm Isolation
## 1. Scope
Triggered upon alarm ALARM-UPS-801 (Battery String Thermal Runaway) or STS Auto-Transfer inhibit.
## 2. Execution Steps
- Step 1: Don NFPA 70E Category 2 Arc Flash gear before entering the UPS battery enclosure.
- Step 2: Verify STS-1 input voltage and phase synchronicity across Line 1 and Line 2.
- Step 3: Isolate Breaker DC-CB-4 to disconnect the faulted battery string from the DC bus.
- Step 4: Transfer UPS output to Static Bypass mode to prevent downstream critical load drop.
- Step 5: Notify High Voltage Electrical Contractor for rapid battery cell replacement.
""",
                procedure_card=ProcedureCard(
                    sop_id="DC-SOP-UPS-003",
                    domain="data_center",
                    title="Static Transfer Switch and UPS Battery String Alarm Isolation",
                    abstract="Isolation procedure for UPS battery string fault or STS auto-transfer inhibit under electrical alarm."
                ),
                entities=[
                    Entity(name="ALARM-UPS-801", category="alarm", attributes={"description": "Battery String Thermal Runaway"}),
                    Entity(name="STS-1", category="equipment", attributes={"type": "Static Transfer Switch"}),
                    Entity(name="DC-CB-4", category="equipment", attributes={"type": "DC Circuit Breaker"}),
                    Entity(name="NFPA 70E PPE", category="parameter", attributes={"category": "2"}),
                    Entity(name="UPS Static Bypass", category="equipment", attributes={"mode": "Bypass"})
                ],
                causal_states=[
                    "Battery Thermal Runaway", "DC Bus Overvoltage", "Battery String Fault", "UPS Phase Desync", "Inverter Trip"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Battery Thermal Runaway", effect_state="Battery String Fault", relation_type="causes"),
                    CausalRelation(cause_state="Battery String Fault", effect_state="DC Bus Overvoltage", relation_type="causes"),
                    CausalRelation(cause_state="DC Bus Overvoltage", effect_state="Inverter Trip", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="DC-SOP-UPS-003",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Don NFPA 70E Category 2 Arc Flash gear before entering the UPS battery enclosure.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Verify STS-1 input voltage and phase synchronicity across Line 1 and Line 2.", step_type="condition"),
                        SOPStep(step_id="s3", order=3, text="Isolate Breaker DC-CB-4 to disconnect the faulted battery string from the DC bus.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Transfer UPS output to Static Bypass mode to prevent downstream critical load drop.", step_type="action"),
                        SOPStep(step_id="s5", order=5, text="Notify High Voltage Electrical Contractor for rapid battery cell replacement.", step_type="action", tool_trigger="call_agent")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="DC-SOP-NET-008",
                domain="data_center",
                document_id="DC-SOP-NET-008",
                version="1.5",
                title="Top-of-Rack Switch BGP Flapping and Spine Partition Recovery",
                scope="Data Hall Network Fabric, ToR Switches TOR-1A to TOR-48B, Spine BGP sessions",
                pre_action_requirements=["Console serial cable", "NetOps authorization ticket", "TACACS+ credentials"],
                raw_markdown="""# Top-of-Rack Switch BGP Flapping and Spine Partition Recovery
## 1. Scope
Triggered when BGP peering state flaps >3 times in 5 minutes (ALARM-NET-BGP-602).
## 2. Action Sequence
- Step 1: Dampen BGP route advertisements on ToR switch using route-map dampening.
- Step 2: Inspect optical transceiver receive power (Rx dBm) on uplinks QSFP-1 and QSFP-2.
- Step 3: Clean and reseat LC fiber patch cables if optical attenuation exceeds -14 dBm.
- Step 4: Clear BGP soft neighbor session and observe EVPN-VXLAN route convergence.
- Step 5: Confirm server network gateway ping latency drops below 0.5 ms.
""",
                procedure_card=ProcedureCard(
                    sop_id="DC-SOP-NET-008",
                    domain="data_center",
                    title="Top-of-Rack Switch BGP Flapping and Spine Partition Recovery",
                    abstract="Stabilization procedure for BGP flapping on top-of-rack switches and optical link degradation."
                ),
                entities=[
                    Entity(name="ALARM-NET-BGP-602", category="alarm", attributes={"type": "BGP Flapping"}),
                    Entity(name="QSFP-1", category="equipment", attributes={"type": "Optical Transceiver"}),
                    Entity(name="QSFP-2", category="equipment", attributes={"type": "Optical Transceiver"}),
                    Entity(name="EVPN-VXLAN", category="parameter", attributes={"protocol": "Overlay"}),
                    Entity(name="NetOps", category="role", attributes={"team": "Network Operations"})
                ],
                causal_states=[
                    "BGP Session Flap", "Optical Attenuation", "Dirty Fiber Endface", "Spine Route Partition"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Dirty Fiber Endface", effect_state="Optical Attenuation", relation_type="causes"),
                    CausalRelation(cause_state="Optical Attenuation", effect_state="BGP Session Flap", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="DC-SOP-NET-008",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Dampen BGP route advertisements on ToR switch using route-map dampening.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Inspect optical transceiver receive power on uplinks QSFP-1 and QSFP-2.", step_type="condition"),
                        SOPStep(step_id="s3", order=3, text="Clean and reseat LC fiber patch cables if attenuation exceeds -14 dBm.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Clear BGP soft neighbor session and observe EVPN-VXLAN route convergence.", step_type="action"),
                        SOPStep(step_id="s5", order=5, text="Confirm server network gateway ping latency drops below 0.5 ms.", step_type="condition")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            )
        ]

    @staticmethod
    def get_liquid_cooling_sops() -> List[SOPDocument]:
        return [
            SOPDocument(
                sop_id="LC-SOP-LEAK-001",
                domain="liquid_cooling",
                document_id="LC-SOP-LEAK-001",
                version="3.4",
                title="Secondary Loop Coolant Leak Detection and Fast Containment",
                scope="Direct-to-Chip (D2C) Cold Plates, CDU secondary manifold, quick-disconnect fittings",
                pre_action_requirements=["Spill containment kit", "Safety goggles & nitrile gloves", "Coolant absorption pads"],
                raw_markdown="""# Secondary Loop Coolant Leak Detection and Fast Containment
## 1. Scope
Mandatory procedure when optical rope sensor triggers ALARM-LEAK-101 or CDU secondary expansion tank drops >5% in 60s.
## 2. Emergency Mitigation
- Step 1: Immediately trip CDU secondary supply pump via emergency SCADA override button.
- Step 2: Close manual isolation ball valves V-201 and V-202 on the rack manifold supply/return headers.
- Step 3: Deploy dielectric absorbent pads around server chassis base and drip trays.
- Step 4: Check server node board moisture sensors; if wet, power down chassis blade immediately.
- Step 5: Depressurize manifold by opening purge drain valve V-209 into sealed containment carboy.
""",
                procedure_card=ProcedureCard(
                    sop_id="LC-SOP-LEAK-001",
                    domain="liquid_cooling",
                    title="Secondary Loop Coolant Leak Detection and Fast Containment",
                    abstract="Immediate containment protocol for direct-to-chip coolant leaks to protect compute blades from fluid damage."
                ),
                entities=[
                    Entity(name="ALARM-LEAK-101", category="alarm", attributes={"type": "Optical Rope Leak"}),
                    Entity(name="CDU", category="equipment", attributes={"type": "Cooling Distribution Unit"}),
                    Entity(name="V-201", category="equipment", attributes={"type": "Supply Ball Valve"}),
                    Entity(name="V-202", category="equipment", attributes={"type": "Return Ball Valve"}),
                    Entity(name="V-209", category="equipment", attributes={"type": "Purge Drain Valve"}),
                    Entity(name="Dielectric Pads", category="equipment", attributes={"type": "Absorbent"})
                ],
                causal_states=[
                    "Coolant Leakage", "Manifold Pressure Drop", "Expansion Tank Depletion", "Fitting Seal Rupture", "Chassis Fluid Ingress"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Fitting Seal Rupture", effect_state="Coolant Leakage", relation_type="causes"),
                    CausalRelation(cause_state="Coolant Leakage", effect_state="Manifold Pressure Drop", relation_type="causes"),
                    CausalRelation(cause_state="Coolant Leakage", effect_state="Expansion Tank Depletion", relation_type="causes"),
                    CausalRelation(cause_state="Coolant Leakage", effect_state="Chassis Fluid Ingress", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="LC-SOP-LEAK-001",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Trip CDU secondary supply pump via emergency SCADA override button.", step_type="action", tool_trigger="dcim_scada_controller"),
                        SOPStep(step_id="s2", order=2, text="Close manual isolation ball valves V-201 and V-202 on the rack manifold supply/return headers.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Deploy dielectric absorbent pads around server chassis base and drip trays.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Check server node board moisture sensors; if wet, power down chassis blade immediately.", step_type="condition"),
                        SOPStep(step_id="s5", order=5, text="Depressurize manifold by opening purge drain valve V-209 into sealed containment carboy.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="LC-SOP-LEAK-002",
                domain="liquid_cooling",
                document_id="LC-SOP-LEAK-002",
                version="2.0",
                title="Primary Facility Chilled Water Supply Pipe Rupture and Header Isolation",
                scope="Facility primary chilled water header, 10-inch butterfly valves, building central riser",
                pre_action_requirements=["Hydraulic valve torque wrench", "Submersible sump pump", "Facilities Incident Commander"],
                raw_markdown="""# Primary Facility Chilled Water Supply Pipe Rupture and Header Isolation
## 1. Scope
Triggered by major flow meter deviation (>500 GPM loss) or primary pipe seismic shear rupture (ALARM-FAC-LEAK-900).
## 2. Emergency Mitigation
- Step 1: Signal central plant automation to shut motorized primary isolation valves PV-401 and PV-402.
- Step 2: Trip primary distribution chillers CH-1 and CH-2 to prevent dead-head pumping.
- Step 3: Engage basement flood sump pumps SP-1 and SP-2 to clear flooded mechanical trench.
- Step 4: Isolate facility makeup water feed to halt ongoing water loss.
""",
                procedure_card=ProcedureCard(
                    sop_id="LC-SOP-LEAK-002",
                    domain="liquid_cooling",
                    title="Primary Facility Chilled Water Supply Pipe Rupture and Header Isolation",
                    abstract="Facility-level isolation protocol for massive primary chilled water loop pipe ruptures."
                ),
                entities=[
                    Entity(name="ALARM-FAC-LEAK-900", category="alarm", attributes={"type": "Primary Pipe Rupture"}),
                    Entity(name="PV-401", category="equipment", attributes={"type": "Primary Butterfly Valve"}),
                    Entity(name="PV-402", category="equipment", attributes={"type": "Primary Butterfly Valve"}),
                    Entity(name="SP-1", category="equipment", attributes={"type": "Sump Pump"}),
                    Entity(name="CH-1", category="equipment", attributes={"type": "Primary Chiller"})
                ],
                causal_states=[
                    "Primary Pipe Rupture", "Seismic Shear", "Massive Flooding", "Chiller Dead-Head Trip"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Seismic Shear", effect_state="Primary Pipe Rupture", relation_type="causes"),
                    CausalRelation(cause_state="Primary Pipe Rupture", effect_state="Massive Flooding", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="LC-SOP-LEAK-002",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Signal central plant automation to shut motorized primary isolation valves PV-401 and PV-402.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Trip primary distribution chillers CH-1 and CH-2 to prevent dead-head pumping.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Engage basement flood sump pumps SP-1 and SP-2 to clear flooded mechanical trench.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Isolate facility makeup water feed to halt ongoing water loss.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4")]
                )
            ),
            SOPDocument(
                sop_id="LC-SOP-CAVIT-004",
                domain="liquid_cooling",
                document_id="LC-SOP-CAVIT-004",
                version="1.8",
                title="CDU Pump Cavitation and Air Ingress De-aeration Protocol",
                scope="Primary and secondary liquid cooling distribution unit circulation pumps",
                pre_action_requirements=["Nitrogen de-aeration kit", "Differential pressure gauge", "Glycol refractometer"],
                raw_markdown="""# CDU Pump Cavitation and Air Ingress De-aeration Protocol
## 1. Scope
Triggered by abnormal pump noise (rattling gravel sound), high vibration (>4.5 mm/s), or suction pressure below 12 PSI.
## 2. Action Sequence
- Step 1: Throttle pump VFD speed down by 20% to prevent impeller erosion from vapor collapse.
- Step 2: Inspect suction side strainer S-101 for particulate clogging and debris accumulation.
- Step 3: Open automatic micro-bubble air separator vent valve AV-12 to vent entrapped nitrogen/air.
- Step 4: Verify positive suction head (NPSHa); inject PG25 coolant mixture until suction gauge reads >18 PSI.
- Step 5: Ramp pump back to nominal RPM and monitor acoustic spectrum for 15 minutes.
""",
                procedure_card=ProcedureCard(
                    sop_id="LC-SOP-CAVIT-004",
                    domain="liquid_cooling",
                    title="CDU Pump Cavitation and Air Ingress De-aeration Protocol",
                    abstract="Remediation procedure for pump cavitation noise, low suction pressure, and micro-bubble entrainment."
                ),
                entities=[
                    Entity(name="S-101", category="equipment", attributes={"type": "Suction Strainer"}),
                    Entity(name="AV-12", category="equipment", attributes={"type": "Air Vent Valve"}),
                    Entity(name="PG25", category="parameter", attributes={"coolant": "Propylene Glycol 25%"}),
                    Entity(name="NPSHa", category="parameter", attributes={"requirement": ">18 PSI"}),
                    Entity(name="Vibration", category="parameter", attributes={"threshold": "4.5 mm/s"}),
                    Entity(name="CDU Pump", category="equipment", attributes={"type": "Circulation Pump"})
                ],
                causal_states=[
                    "Pump Cavitation", "Vapor Bubble Collapse", "Low Suction Pressure", "Strainer Clogging", "Entrapped Air"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Strainer Clogging", effect_state="Low Suction Pressure", relation_type="causes"),
                    CausalRelation(cause_state="Low Suction Pressure", effect_state="Vapor Bubble Collapse", relation_type="causes"),
                    CausalRelation(cause_state="Vapor Bubble Collapse", effect_state="Pump Cavitation", relation_type="causes"),
                    CausalRelation(cause_state="Entrapped Air", effect_state="Pump Cavitation", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="LC-SOP-CAVIT-004",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Throttle pump VFD speed down by 20% to prevent impeller erosion from vapor collapse.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Inspect suction side strainer S-101 for particulate clogging and debris accumulation.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Open automatic micro-bubble air separator vent valve AV-12 to vent entrapped nitrogen/air.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Verify positive suction head (NPSHa); inject PG25 coolant mixture until suction gauge reads >18 PSI.", step_type="condition"),
                        SOPStep(step_id="s5", order=5, text="Ramp pump back to nominal RPM and monitor acoustic spectrum for 15 minutes.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            )
        ]

    @staticmethod
    def get_building_management_sops() -> List[SOPDocument]:
        return [
            SOPDocument(
                sop_id="BM-SOP-HVAC-002",
                domain="building_management",
                document_id="BM-SOP-HVAC-002",
                version="5.0",
                title="Centrifugal Chiller Surge and Compressor Surge Protection Protocol",
                scope="Central Chiller Plant, Chillers CH-1 through CH-4, Variable Orifice and Vane Control",
                pre_action_requirements=["BMS Master workstation", "Chiller maintenance radio", "Ear protection"],
                raw_markdown="""# Centrifugal Chiller Surge and Compressor Surge Protection Protocol
## 1. Scope
Triggered upon compressor surge cycle sound (whooshing/thumping) or alarm ALARM-CH-SURGE-401.
## 2. Action Sequence
- Step 1: Immediately verify condenser water entering temperature (CWET); if >85°F, engage auxiliary cooling tower fan CT-3.
- Step 2: Command the pre-rotation guide vanes (PRGV) to drive towards 30% minimum position to stabilize refrigerant gas flow.
- Step 3: Open the hot-gas bypass valve (HGBP) to increase suction mass flow and unload compressor impellers.
- Step 4: Verify lift pressure across compressor; confirm discharge pressure stabilizes below 140 PSIG.
- Step 5: If surge cycles exceed 3 events in 10 minutes, execute controlled chiller shutdown to protect magnetic bearings.
""",
                procedure_card=ProcedureCard(
                    sop_id="BM-SOP-HVAC-002",
                    domain="building_management",
                    title="Centrifugal Chiller Surge and Compressor Surge Protection Protocol",
                    abstract="Emergency anti-surge stabilization protocol for central plant centrifugal chillers to prevent magnetic bearing failure."
                ),
                entities=[
                    Entity(name="ALARM-CH-SURGE-401", category="alarm", attributes={"type": "Chiller Compressor Surge"}),
                    Entity(name="CH-1", category="equipment", attributes={"type": "Centrifugal Chiller"}),
                    Entity(name="CT-3", category="equipment", attributes={"type": "Cooling Tower Fan"}),
                    Entity(name="PRGV", category="equipment", attributes={"type": "Pre-Rotation Guide Vanes"}),
                    Entity(name="HGBP", category="equipment", attributes={"type": "Hot Gas Bypass Valve"}),
                    Entity(name="CWET", category="parameter", attributes={"threshold": "85F"})
                ],
                causal_states=[
                    "Compressor Surge", "High Condenser Head Pressure", "High CWET", "Low Refrigerant Flow", "Magnetic Bearing Overheat"
                ],
                causal_relations=[
                    CausalRelation(cause_state="High CWET", effect_state="High Condenser Head Pressure", relation_type="causes"),
                    CausalRelation(cause_state="High Condenser Head Pressure", effect_state="Compressor Surge", relation_type="causes"),
                    CausalRelation(cause_state="Low Refrigerant Flow", effect_state="Compressor Surge", relation_type="causes"),
                    CausalRelation(cause_state="Compressor Surge", effect_state="Magnetic Bearing Overheat", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="BM-SOP-HVAC-002",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Immediately verify condenser water entering temperature (CWET); if >85°F, engage auxiliary cooling tower fan CT-3.", step_type="condition"),
                        SOPStep(step_id="s2", order=2, text="Command pre-rotation guide vanes (PRGV) to drive towards 30% minimum position.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Open hot-gas bypass valve (HGBP) to increase suction mass flow and unload impellers.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Verify lift pressure across compressor; confirm discharge pressure stabilizes below 140 PSIG.", step_type="condition"),
                        SOPStep(step_id="s5", order=5, text="If surge cycles exceed 3 events in 10 minutes, execute controlled chiller shutdown.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="BM-SOP-FIRE-007",
                domain="building_management",
                document_id="BM-SOP-FIRE-007",
                version="4.1",
                title="Fire Suppression Waterflow Alarm and Pre-Action Sprinkler Response",
                scope="Facility Zone Sprinkler Risers, Pre-Action Deluge Valves, Fire Alarm Control Panel (FACP)",
                pre_action_requirements=["Master FACP Key", "Zone locator map", "Facilities 2-way walkie-talkie"],
                raw_markdown="""# Fire Suppression Waterflow Alarm and Pre-Action Sprinkler Response
## 1. Scope
Mandatory procedure upon FACP Waterflow Switch activation (ALARM-WF-202) in Electrical/Server rooms.
## 2. Execution Steps
- Step 1: Check FACP annunciator panel to identify the active Zone and Cross-Zoned smoke detector status.
- Step 2: Dispatch Security / Facilities patrol to visual verify existence of actual flame or smoke in the zone.
- Step 3: If false alarm due to pipe pressure surge, press FACP Silence and hold Pre-Action Deluge Manual Abort Switch.
- Step 4: If verified fire, ensure Pre-Action Valve PAV-01 solenoid trips and verify fire pump start confirmation.
- Step 5: Coordinate with Municipal Fire Department (Call 911) and initiate building egress public announcement.
""",
                procedure_card=ProcedureCard(
                    sop_id="BM-SOP-FIRE-007",
                    domain="building_management",
                    title="Fire Suppression Waterflow Alarm and Pre-Action Sprinkler Response",
                    abstract="Verification and response procedure for pre-action sprinkler waterflow alarms to prevent accidental water discharge."
                ),
                entities=[
                    Entity(name="ALARM-WF-202", category="alarm", attributes={"type": "Waterflow Switch Alarm"}),
                    Entity(name="FACP", category="equipment", attributes={"type": "Fire Alarm Control Panel"}),
                    Entity(name="PAV-01", category="equipment", attributes={"type": "Pre-Action Deluge Valve"}),
                    Entity(name="Fire Pump", category="equipment", attributes={"type": "Diesel Booster Pump"}),
                    Entity(name="Security", category="role", attributes={"team": "Patrol"})
                ],
                causal_states=[
                    "Waterflow Switch Tripped", "Pipe Pressure Surge", "Smoke Detector Cross-Zone", "Accidental Water Discharge", "Structure Fire"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Pipe Pressure Surge", effect_state="Waterflow Switch Tripped", relation_type="causes"),
                    CausalRelation(cause_state="Smoke Detector Cross-Zone", effect_state="Waterflow Switch Tripped", relation_type="causes"),
                    CausalRelation(cause_state="Waterflow Switch Tripped", effect_state="Accidental Water Discharge", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="BM-SOP-FIRE-007",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Check FACP annunciator panel to identify active Zone and Cross-Zoned smoke detector status.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Dispatch Security / Facilities patrol to visual verify existence of flame or smoke.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="If false alarm due to pressure surge, press FACP Silence and hold Pre-Action Abort Switch.", step_type="condition"),
                        SOPStep(step_id="s4", order=4, text="If verified fire, ensure Pre-Action Valve PAV-01 solenoid trips and verify fire pump starts.", step_type="action"),
                        SOPStep(step_id="s5", order=5, text="Coordinate with Municipal Fire Department (Call 911) and initiate evacuation broadcast.", step_type="action", tool_trigger="call_agent")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="BM-SOP-FIRE-008",
                domain="building_management",
                document_id="BM-SOP-FIRE-008",
                version="3.0",
                title="Clean Agent FM-200 and Novec Gaseous Fire Extinguishing Release Inhibit",
                scope="Data Hall Clean Agent Gas cylinders, pneumatic release dampers, abort switch stations",
                pre_action_requirements=["Key to Manual Release Station", "SCBA Self-Contained Breathing Apparatus"],
                raw_markdown="""# Clean Agent FM-200 and Novec Gaseous Fire Extinguishing Release Inhibit
## 1. Scope
Immediate protocol when accidental gas countdown timer (30-second warning buzzer) initiates in occupied room.
## 2. Execution Steps
- Step 1: Immediately press and continuously HOLD the yellow Clean Agent Manual Abort button.
- Step 2: Ensure room occupants evacuate while keeping abort switch depressed.
- Step 3: Verify HVAC motorized isolation dampers close to retain agent boundary.
- Step 4: Reset the releasing panel using technician bypass key before releasing abort switch.
""",
                procedure_card=ProcedureCard(
                    sop_id="BM-SOP-FIRE-008",
                    domain="building_management",
                    title="Clean Agent FM-200 and Novec Gaseous Fire Extinguishing Release Inhibit",
                    abstract="Immediate hold-abort protocol to inhibit accidental gaseous clean-agent discharge in occupied facilities."
                ),
                entities=[
                    Entity(name="FM-200", category="parameter", attributes={"type": "Gaseous Clean Agent"}),
                    Entity(name="Novec", category="parameter", attributes={"type": "Clean Agent"}),
                    Entity(name="Manual Abort Switch", category="equipment", attributes={"color": "Yellow"}),
                    Entity(name="SCBA", category="equipment", attributes={"type": "Breathing Gear"}),
                    Entity(name="Isolation Damper", category="equipment", attributes={"type": "HVAC Fire Damper"})
                ],
                causal_states=[
                    "Gas Countdown Buzzer", "Accidental Discharge Risk", "Asphyxiation Hazard", "False Optical Sensor Trip"
                ],
                causal_relations=[
                    CausalRelation(cause_state="False Optical Sensor Trip", effect_state="Gas Countdown Buzzer", relation_type="causes"),
                    CausalRelation(cause_state="Gas Countdown Buzzer", effect_state="Accidental Discharge Risk", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="BM-SOP-FIRE-008",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Immediately press and continuously HOLD the yellow Clean Agent Manual Abort button.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Ensure room occupants evacuate while keeping abort switch depressed.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Verify HVAC motorized isolation dampers close to retain agent boundary.", step_type="condition"),
                        SOPStep(step_id="s4", order=4, text="Reset releasing panel using technician bypass key before releasing abort switch.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4")]
                )
            )
        ]

    @staticmethod
    def get_airline_services_sops() -> List[SOPDocument]:
        return [
            SOPDocument(
                sop_id="AIR-SOP-UPSET-001",
                domain="airline_services",
                document_id="AIR-QRH-737-08",
                version="6.2",
                title="Aircraft Upset Recovery and Unusual Attitude Recovery Technique",
                scope="Flight Deck Crew, All flight phases, Nose-High or Nose-Low bank exceedance >45 deg",
                pre_action_requirements=["Autopilot disconnect", "Autothrottle disconnect", "PF/PM Crew Coordination"],
                raw_markdown="""# Aircraft Upset Recovery and Unusual Attitude Recovery Technique
## 1. Scope
Immediate pilot action checklist when aircraft encounters severe turbulence, wake vortex, or pitch/roll upset.
## 2. Actionable Procedure
- Step 1: Disconnect Autopilot and Autothrottle immediately; push AP/AT disconnect switches on yoke.
- Step 2: Recognize and confirm the upset condition on Primary Flight Display (PFD) attitude indicator.
- Step 3: For Nose-High recovery: apply nose-down elevator, adjust thrust as needed, roll wings level.
- Step 4: For Nose-Low recovery: unload G-forces, roll wings level to horizon, smoothly apply nose-up elevator.
- Step 5: Check airspeed and pitch trim; re-engage flight automation once stabilized in level flight.
""",
                procedure_card=ProcedureCard(
                    sop_id="AIR-SOP-UPSET-001",
                    domain="airline_services",
                    title="Aircraft Upset Recovery and Unusual Attitude Recovery Technique",
                    abstract="Standard flight crew procedure for recovering from severe nose-high or nose-low aircraft upset and unusual attitude."
                ),
                entities=[
                    Entity(name="Autopilot", category="equipment", attributes={"switch": "AP Disconnect"}),
                    Entity(name="Autothrottle", category="equipment", attributes={"switch": "AT Disconnect"}),
                    Entity(name="PFD", category="equipment", attributes={"instrument": "Primary Flight Display"}),
                    Entity(name="Elevator", category="equipment", attributes={"flight_control": "Pitch"}),
                    Entity(name="Pitch Trim", category="parameter", attributes={"type": "Stabilizer Trim"}),
                    Entity(name="Pilot Flying", category="role", attributes={"crew": "PF"})
                ],
                causal_states=[
                    "Aircraft Upset", "Severe Wake Vortex", "Stall Imminent", "Nose-High Pitch Exceedance", "Nose-Low Roll Exceedance"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Severe Wake Vortex", effect_state="Aircraft Upset", relation_type="causes"),
                    CausalRelation(cause_state="Aircraft Upset", effect_state="Nose-High Pitch Exceedance", relation_type="causes"),
                    CausalRelation(cause_state="Nose-High Pitch Exceedance", effect_state="Stall Imminent", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="AIR-SOP-UPSET-001",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Disconnect Autopilot and Autothrottle immediately; push AP/AT disconnect switches on yoke.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Recognize and confirm the upset condition on Primary Flight Display (PFD) attitude indicator.", step_type="condition"),
                        SOPStep(step_id="s3", order=3, text="For Nose-High: apply nose-down elevator, adjust thrust as needed, roll wings level.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="For Nose-Low: unload G-forces, roll wings level to horizon, smoothly apply nose-up elevator.", step_type="action"),
                        SOPStep(step_id="s5", order=5, text="Check airspeed and pitch trim; re-engage flight automation once stabilized in level flight.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="AIR-SOP-UPSET-002",
                domain="airline_services",
                document_id="AIR-QRH-737-09",
                version="4.0",
                title="Stick Shaker Activation and Aerodynamic Stall Recovery Procedure",
                scope="Flight Deck Crew, Approach to stall, Stick Shaker rattling, buffeting onset",
                pre_action_requirements=["Pitch trim disconnect", "Control column positive forward movement"],
                raw_markdown="""# Stick Shaker Activation and Aerodynamic Stall Recovery Procedure
## 1. Scope
Mandatory pilot response upon Stick Shaker actuation or aerodynamic buffet warning.
## 2. Action Sequence
- Step 1: Immediately disconnect Autopilot and apply nose-down pitch control column firmly.
- Step 2: Roll wings level to ensure shortest path to lift vector alignment.
- Step 3: Advance thrust levers smoothly to maximum go-around thrust (TOGA).
- Step 4: Retract speedbrakes if deployed; do NOT change gear or flap configuration until stall is broken.
- Step 5: Accelerate above VREF+20 and recover altitude once positive climb is established.
""",
                procedure_card=ProcedureCard(
                    sop_id="AIR-SOP-UPSET-002",
                    domain="airline_services",
                    title="Stick Shaker Activation and Aerodynamic Stall Recovery Procedure",
                    abstract="Immediate stall barrier recovery procedure when stick shaker activates or aerodynamic buffet occurs."
                ),
                entities=[
                    Entity(name="Stick Shaker", category="alarm", attributes={"warning": "Stall warning"}),
                    Entity(name="TOGA", category="parameter", attributes={"thrust": "Takeoff Go-Around"}),
                    Entity(name="VREF", category="parameter", attributes={"airspeed": "Reference"}),
                    Entity(name="Speedbrakes", category="equipment", attributes={"flight_control": "Spoilers"}),
                    Entity(name="Control Column", category="equipment", attributes={"flight_control": "Yoke"})
                ],
                causal_states=[
                    "Stick Shaker Rattling", "Aerodynamic Buffet", "Loss of Airspeed", "Angle of Attack Exceedance"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Loss of Airspeed", effect_state="Angle of Attack Exceedance", relation_type="causes"),
                    CausalRelation(cause_state="Angle of Attack Exceedance", effect_state="Stick Shaker Rattling", relation_type="causes"),
                    CausalRelation(cause_state="Stick Shaker Rattling", effect_state="Aerodynamic Buffet", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="AIR-SOP-UPSET-002",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Immediately disconnect Autopilot and apply nose-down pitch control column firmly.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Roll wings level to ensure shortest path to lift vector alignment.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Advance thrust levers smoothly to maximum go-around thrust (TOGA).", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Retract speedbrakes if deployed; do not change gear or flap configuration.", step_type="condition"),
                        SOPStep(step_id="s5", order=5, text="Accelerate above VREF+20 and recover altitude once positive climb is established.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            ),
            SOPDocument(
                sop_id="AIR-SOP-CABIN-009",
                domain="airline_services",
                document_id="AIR-QRH-737-12",
                version="5.3",
                title="Rapid Cabin Depressurization and Emergency Descent Procedure",
                scope="Flight Deck & Cabin Crew, Cabin Altitude Warning (>10,000 ft), Structural Pressure Loss",
                pre_action_requirements=["Don oxygen masks 100%", "Establish crew interphone", "Transponder Squawk 7700"],
                raw_markdown="""# Rapid Cabin Depressurization and Emergency Descent Procedure
## 1. Scope
Checklist executed when Cabin Altitude horn sounds or cabin pressure differential drops rapidly towards zero.
## 2. Execution Protocol
- Step 1: Don flight crew oxygen masks (100% Emergency position) and establish pilot interphone communication.
- Step 2: Passenger Signs ON; deploy passenger cabin oxygen masks manually via overhead toggle.
- Step 3: Initiate Rapid Emergency Descent: turn heading off airway, deploy Speedbrakes fully, set MCP altitude to 10,000 ft or MEA.
- Step 4: Set Transponder to Squawk 7700; broadcast Mayday call on ATC emergency guard frequency 121.5 MHz.
- Step 5: Level off at 10,000 feet or Minimum Enroute Altitude; assess structural integrity and divert to nearest airport.
""",
                procedure_card=ProcedureCard(
                    sop_id="AIR-SOP-CABIN-009",
                    domain="airline_services",
                    title="Rapid Cabin Depressurization and Emergency Descent Procedure",
                    abstract="Immediate pilot emergency descent checklist upon rapid cabin pressure loss and cabin altitude alarm."
                ),
                entities=[
                    Entity(name="Oxygen Mask 100%", category="equipment", attributes={"pressure": "Positive"}),
                    Entity(name="Squawk 7700", category="parameter", attributes={"transponder": "Emergency"}),
                    Entity(name="Speedbrakes", category="equipment", attributes={"flight_control": "Flight Spoilers"}),
                    Entity(name="Guard 121.5", category="parameter", attributes={"frequency": "Emergency VHF"}),
                    Entity(name="Air Traffic Control", category="role", attributes={"radio": "ATC"})
                ],
                causal_states=[
                    "Cabin Depressurization", "Outflow Valve Failure", "Hull Breach", "Hypoxia Danger", "Cabin Altitude Horn"
                ],
                causal_relations=[
                    CausalRelation(cause_state="Outflow Valve Failure", effect_state="Cabin Depressurization", relation_type="causes"),
                    CausalRelation(cause_state="Hull Breach", effect_state="Cabin Depressurization", relation_type="causes"),
                    CausalRelation(cause_state="Cabin Depressurization", effect_state="Cabin Altitude Horn", relation_type="causes"),
                    CausalRelation(cause_state="Cabin Depressurization", effect_state="Hypoxia Danger", relation_type="causes")
                ],
                flow_graph=FlowGraphData(
                    sop_id="AIR-SOP-CABIN-009",
                    steps=[
                        SOPStep(step_id="s1", order=1, text="Don flight crew oxygen masks (100% Emergency position) and establish pilot interphone communication.", step_type="action"),
                        SOPStep(step_id="s2", order=2, text="Passenger Signs ON; deploy passenger cabin oxygen masks manually via overhead toggle.", step_type="action"),
                        SOPStep(step_id="s3", order=3, text="Initiate Rapid Emergency Descent: turn heading off airway, deploy Speedbrakes fully, set MCP altitude to 10,000 ft.", step_type="action"),
                        SOPStep(step_id="s4", order=4, text="Set Transponder to Squawk 7700; broadcast Mayday call on ATC emergency guard frequency 121.5 MHz.", step_type="action", tool_trigger="call_agent"),
                        SOPStep(step_id="s5", order=5, text="Level off at 10,000 feet or Minimum Enroute Altitude; assess structural integrity and divert.", step_type="action")
                    ],
                    edges=[("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5")]
                )
            )
        ]

    @classmethod
    def get_all_sops(cls) -> List[SOPDocument]:
        return (
            cls.get_data_center_sops() +
            cls.get_liquid_cooling_sops() +
            cls.get_building_management_sops() +
            cls.get_airline_services_sops()
        )

    @staticmethod
    def get_evaluation_queries() -> List[Dict[str, Any]]:
        return [
            # Data Center
            {
                "query_id": "DC-Q1",
                "domain": "data_center",
                "query_type": "entity",
                "query": "How to initiate a hard server reboot using iDRAC or ILO console?",
                "target_sop_id": "DC-SOP-CRIT-005.A"
            },
            {
                "query_id": "DC-Q2",
                "domain": "data_center",
                "query_type": "causal",
                "query": "Why does cold aisle containment leakage cause PUE spike and thermal overload?",
                "target_sop_id": "DC-SOP-PUE-012"
            },
            {
                "query_id": "DC-Q3",
                "domain": "data_center",
                "query_type": "flow",
                "query": "What are the step-by-step instructions to isolate a faulted battery string when ALARM-UPS-801 triggers?",
                "target_sop_id": "DC-SOP-UPS-003"
            },
            {
                "query_id": "DC-Q4",
                "domain": "data_center",
                "query_type": "flow",
                "query": "What is the procedure for handling an unrecoverable host blackout and node outage?",
                "target_sop_id": "DC-SOP-CRIT-005.A"
            },
            {
                "query_id": "DC-Q5",
                "domain": "data_center",
                "query_type": "entity",
                "query": "What steps should be taken when optical attenuation triggers ALARM-NET-BGP-602 on QSFP-1?",
                "target_sop_id": "DC-SOP-NET-008"
            },
            {
                "query_id": "DC-Q6",
                "domain": "data_center",
                "query_type": "flow",
                "query": "How to perform live migration and hypervisor evacuation during cluster quorum failure DR-99?",
                "target_sop_id": "DC-SOP-CRIT-005.B"
            },

            # Liquid Cooling
            {
                "query_id": "LC-Q1",
                "domain": "liquid_cooling",
                "query_type": "entity",
                "query": "Which ball valves V-201 and V-202 must be isolated during ALARM-LEAK-101?",
                "target_sop_id": "LC-SOP-LEAK-001"
            },
            {
                "query_id": "LC-Q2",
                "domain": "liquid_cooling",
                "query_type": "causal",
                "query": "Why is pump cavitation causing high vibration and rattling noise in the liquid cooling loop?",
                "target_sop_id": "LC-SOP-CAVIT-004"
            },
            {
                "query_id": "LC-Q3",
                "domain": "liquid_cooling",
                "query_type": "flow",
                "query": "How do I perform emergency containment of a secondary loop coolant leak?",
                "target_sop_id": "LC-SOP-LEAK-001"
            },
            {
                "query_id": "LC-Q4",
                "domain": "liquid_cooling",
                "query_type": "flow",
                "query": "What are the de-aeration steps to purge entrapped air and restore suction pressure in CDU pump?",
                "target_sop_id": "LC-SOP-CAVIT-004"
            },
            {
                "query_id": "LC-Q5",
                "domain": "liquid_cooling",
                "query_type": "entity",
                "query": "What is the procedure when primary butterfly valves PV-401 and PV-402 must be closed for ALARM-FAC-LEAK-900?",
                "target_sop_id": "LC-SOP-LEAK-002"
            },
            {
                "query_id": "LC-Q6",
                "domain": "liquid_cooling",
                "query_type": "causal",
                "query": "Why did seismic shear cause massive flooding in the central mechanical trench?",
                "target_sop_id": "LC-SOP-LEAK-002"
            },

            # Building Management
            {
                "query_id": "BM-Q1",
                "domain": "building_management",
                "query_type": "entity",
                "query": "What parameter CWET threshold requires engaging cooling tower fan CT-3 during ALARM-CH-SURGE-401?",
                "target_sop_id": "BM-SOP-HVAC-002"
            },
            {
                "query_id": "BM-Q2",
                "domain": "building_management",
                "query_type": "causal",
                "query": "How does high condenser head pressure cause centrifugal compressor surge and magnetic bearing damage?",
                "target_sop_id": "BM-SOP-HVAC-002"
            },
            {
                "query_id": "BM-Q3",
                "domain": "building_management",
                "query_type": "flow",
                "query": "What is the procedure when pre-action sprinkler waterflow alarm ALARM-WF-202 is reported?",
                "target_sop_id": "BM-SOP-FIRE-007"
            },
            {
                "query_id": "BM-Q4",
                "domain": "building_management",
                "query_type": "causal",
                "query": "Why did a pipe pressure surge trigger the waterflow switch alarm?",
                "target_sop_id": "BM-SOP-FIRE-007"
            },
            {
                "query_id": "BM-Q5",
                "domain": "building_management",
                "query_type": "flow",
                "query": "How to hold the manual abort button to prevent FM-200 clean agent gas discharge?",
                "target_sop_id": "BM-SOP-FIRE-008"
            },
            {
                "query_id": "BM-Q6",
                "domain": "building_management",
                "query_type": "entity",
                "query": "Which SCBA equipment and manual abort switch should be used for Novec clean agent alarm?",
                "target_sop_id": "BM-SOP-FIRE-008"
            },

            # Airline Services
            {
                "query_id": "AIR-Q1",
                "domain": "airline_services",
                "query_type": "flow",
                "query": "What is the standard procedure for recovering from an aircraft upset or unusual attitude in flight?",
                "target_sop_id": "AIR-SOP-UPSET-001"
            },
            {
                "query_id": "AIR-Q2",
                "domain": "airline_services",
                "query_type": "flow",
                "query": "What are the emergency descent steps when cabin altitude warning horn triggers?",
                "target_sop_id": "AIR-SOP-CABIN-009"
            },
            {
                "query_id": "AIR-Q3",
                "domain": "airline_services",
                "query_type": "entity",
                "query": "What emergency transponder code Squawk 7700 and radio frequency should be broadcast during rapid descent?",
                "target_sop_id": "AIR-SOP-CABIN-009"
            },
            {
                "query_id": "AIR-Q4",
                "domain": "airline_services",
                "query_type": "causal",
                "query": "Why does wake vortex encounter cause severe aircraft upset and imminent stall?",
                "target_sop_id": "AIR-SOP-UPSET-001"
            },
            {
                "query_id": "AIR-Q5",
                "domain": "airline_services",
                "query_type": "flow",
                "query": "What are the immediate recovery steps when the stick shaker rattles and buffeting begins?",
                "target_sop_id": "AIR-SOP-UPSET-002"
            },
            {
                "query_id": "AIR-Q6",
                "domain": "airline_services",
                "query_type": "causal",
                "query": "Why does angle of attack exceedance trigger the stick shaker warning?",
                "target_sop_id": "AIR-SOP-UPSET-002"
            }
        ]
