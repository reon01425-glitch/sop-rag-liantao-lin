"""
FastAPI Web Dashboard for SOPRAG.
Provides interactive visual interface for:
- Query input & live LLM Router attention weights [w_E, w_C, w_F]
- Multi-view Graph Experts visualization (Entity, Causal, Flow)
- Procedure Cards and Top-K pruning
- Structure-Aware Generation
- Human-in-the-Loop Agent Operational Execution (Figure 7)
- Benchmark comparisons & interactive metrics
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sop_rag.benchmark.dataset import BenchmarkCorpus
from sop_rag.pipeline import SOPRAGPipeline
from sop_rag.experiments.run_benchmarks import BenchmarkExperimentRunner

app = FastAPI(title="SOPRAG Web Dashboard", version="1.0.0")

# Mount benchmark_results for images
if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results", exist_ok=True)
app.mount("/plots", StaticFiles(directory="benchmark_results"), name="plots")

# Initialize pipeline with corpus
pipeline = SOPRAGPipeline()
all_sops = BenchmarkCorpus.get_all_sops()
pipeline.load_documents(all_sops)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    ablation_mode: str = "none"


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_CONTENT


@app.post("/api/query")
async def api_query(req: QueryRequest):
    ablation = None if req.ablation_mode == "none" else req.ablation_mode
    res = pipeline.query(
        query_text=req.query,
        top_k=req.top_k,
        generate_response=True,
        plan_agents=True,
        ablation_mode=ablation
    )

    retrieval_list = []
    for r in res["retrieval_results"]:
        retrieval_list.append({
            "sop_id": r.sop_id,
            "title": r.title,
            "final_score": r.final_score,
            "anchor_score": r.anchor_score,
            "entity_score": r.entity_score,
            "causal_score": r.causal_score,
            "flow_score": r.flow_score,
        })

    router_info = {}
    if res["router_weights"]:
        rw = res["router_weights"]
        router_info = {
            "w_E": rw.w_E,
            "w_C": rw.w_C,
            "w_F": rw.w_F,
            "intent_summary": rw.intent_summary,
            "entities": rw.extracted_entities
        }

    gen_info = {}
    if res["generation"]:
        gen = res["generation"]
        gen_info = {
            "sop_id": gen.sop_id,
            "title": gen.title,
            "response": gen.generated_response,
            "verified_steps": gen.verified_steps,
            "sop_quality_score": gen.sop_quality_score
        }

    return JSONResponse({
        "query": res["query"],
        "latency_ms": res["retrieval_latency_ms"],
        "router": router_info,
        "retrieval": retrieval_list,
        "generation": gen_info,
        "workflow": res["agent_workflow"]
    })


@app.get("/api/sops")
async def get_sops():
    data = []
    for s in all_sops:
        data.append({
            "sop_id": s.sop_id,
            "domain": s.domain,
            "document_id": s.document_id,
            "title": s.title,
            "scope": s.scope,
            "abstract": s.procedure_card.abstract if s.procedure_card else "",
            "entities_count": len(s.entities),
            "causal_states_count": len(s.causal_states),
            "steps_count": len(s.flow_graph.steps) if s.flow_graph else 0
        })
    return JSONResponse(data)


@app.get("/api/sop/{sop_id}/graphs")
async def get_sop_graphs(sop_id: str):
    sop = pipeline.retriever.sops.get(sop_id)
    if not sop:
        return JSONResponse({"error": "SOP not found"}, status_code=404)

    # Entity nodes & edges
    entity_nodes = [{"id": e.name, "label": f"{e.name}\n({e.category})", "group": "entity"} for e in sop.entities]
    entity_nodes.append({"id": sop.sop_id, "label": sop.title[:25] + "...", "group": "sop"})
    entity_edges = [{"from": e.name, "to": sop.sop_id, "label": "associated"} for e in sop.entities]

    # Causal nodes & edges
    causal_nodes = [{"id": st, "label": st, "group": "state"} for st in sop.causal_states]
    causal_nodes.append({"id": sop.sop_id, "label": sop.title[:25] + "...", "group": "sop"})
    causal_edges = [{"from": r.cause_state, "to": r.effect_state, "label": r.relation_type} for r in sop.causal_relations]
    for st in sop.causal_states:
        causal_edges.append({"from": st, "to": sop.sop_id, "label": "resolves_in"})

    # Flow nodes & edges
    flow_nodes = []
    flow_edges = []
    if sop.flow_graph:
        for s in sop.flow_graph.steps:
            flow_nodes.append({
                "id": s.step_id,
                "label": f"Step {s.order}\n{s.text[:35]}...",
                "group": "condition" if s.step_type == "condition" else "action"
            })
        for src, dst in sop.flow_graph.edges:
            flow_edges.append({"from": src, "to": dst})

    return JSONResponse({
        "sop_id": sop.sop_id,
        "title": sop.title,
        "entities": {"nodes": entity_nodes, "edges": entity_edges},
        "causal": {"nodes": causal_nodes, "edges": causal_edges},
        "flow": {"nodes": flow_nodes, "edges": flow_edges}
    })


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SOPRAG: Multi-view Graph Experts Retrieval</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    .glass { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
    .glow-cyan { box-shadow: 0 0 25px rgba(6, 182, 212, 0.25); }
    .network-box { height: 260px; border-radius: 0.75rem; border: 1px solid #334155; background: #0f172a; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur">
    <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <span class="p-2 bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-lg font-black text-xl text-white">SOP</span>
        <div>
          <h1 class="text-lg font-bold tracking-tight">SOPRAG Engine</h1>
          <p class="text-xs text-slate-400">Multi-view Graph Experts for Industrial Standard Operating Procedures</p>
        </div>
      </div>
      <div class="flex items-center space-x-2 text-xs">
        <span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">MoE Gated Retriever Active</span>
        <span class="px-2.5 py-1 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">FastEmbed BGE-small</span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 py-6 space-y-6">
    <!-- Top Query Section -->
    <div class="glass p-5 rounded-2xl glow-cyan space-y-4">
      <label class="block text-sm font-semibold text-slate-300">Operator Query / System Alert</label>
      <div class="flex gap-2">
        <input id="queryInput" type="text"
          value="How to initiate a hard server reboot using iDRAC or ILO console?"
          class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-cyan-400 text-sm shadow-inner"
          placeholder="e.g. ALARM-LEAK-101 ball valve isolation, or PUE spike CRAC failure..." />
        <button onclick="performQuery()" class="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 rounded-xl font-semibold text-sm transition shadow-lg flex items-center gap-2">
          <span>Search SOP</span>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
        </button>
      </div>

      <!-- Quick presets -->
      <div class="flex flex-wrap gap-2 text-xs pt-1">
        <span class="text-slate-500 self-center">Presets:</span>
        <button onclick="setQuery(this.innerText)" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">How to initiate a hard server reboot using iDRAC or ILO console?</button>
        <button onclick="setQuery(this.innerText)" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">Why is pump cavitation causing high vibration and rattling noise in the liquid cooling loop?</button>
        <button onclick="setQuery(this.innerText)" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">What is the standard procedure for recovering from an aircraft upset or unusual attitude in flight?</button>
        <button onclick="setQuery(this.innerText)" class="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">Which ball valves V-201 and V-202 must be isolated during ALARM-LEAK-101?</button>
      </div>
    </div>

    <!-- Dynamic Router Attention Weights (MoE Gating) -->
    <div id="routerCard" class="glass p-5 rounded-2xl space-y-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></span>
          <h2 class="text-md font-bold text-white tracking-wide">Intention-Aware LLM Router (MoE Gating Network)</h2>
        </div>
        <span id="latencyBadge" class="text-xs text-slate-400">Latency: 0 ms</span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Entity Expert -->
        <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center text-xs mb-1">
              <span class="font-semibold text-blue-400">Entity Graph (w_E)</span>
              <span id="weVal" class="font-mono text-blue-300 font-bold">0.00</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 mb-2">
              <div id="weBar" class="bg-blue-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>
          <p class="text-[11px] text-slate-400">Alarms (ALARM101), parameters (temperature/pressure), equipment hardware.</p>
        </div>

        <!-- Causal Expert -->
        <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center text-xs mb-1">
              <span class="font-semibold text-emerald-400">Causal Graph (w_C)</span>
              <span id="wcVal" class="font-mono text-emerald-300 font-bold">0.00</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 mb-2">
              <div id="wcBar" class="bg-emerald-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>
          <p class="text-[11px] text-slate-400">Root-cause diagnosis, symptoms, condition-dependent anomalies.</p>
        </div>

        <!-- Flow Expert -->
        <div class="bg-slate-900/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-center text-xs mb-1">
              <span class="font-semibold text-amber-400">Flow Graph (w_F)</span>
              <span id="wfVal" class="font-mono text-amber-300 font-bold">0.00</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 mb-2">
              <div id="wfBar" class="bg-amber-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
            </div>
          </div>
          <p class="text-[11px] text-slate-400">Actionable step-by-step procedures, topological execution order.</p>
        </div>
      </div>
      <div id="routerIntentSummary" class="text-xs text-slate-300 bg-slate-900/50 p-2.5 rounded-lg border border-slate-800 flex justify-between">
        <span><b>Detected Intent:</b> Initializing...</span>
        <span><b>Extracted Entities:</b> None</span>
      </div>
    </div>

    <!-- Main Content Grid: Left (Ranked Results & Graphs) | Right (Generation & Agent Actions) -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left Column: Retrieval Ranking & Multi-View Subgraphs -->
      <div class="lg:col-span-6 space-y-6">
        <!-- Candidate Cards -->
        <div class="glass p-5 rounded-2xl space-y-3">
          <h2 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Top Retrieved Candidate SOPs</h2>
          <div id="candidatesList" class="space-y-2">
            <!-- Dynamically populated -->
          </div>
        </div>

        <!-- Visual Multi-View Graphs for Selected SOP -->
        <div class="glass p-5 rounded-2xl space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Multi-View Graph Experts Inspector</h2>
            <div class="flex gap-1 text-xs">
              <button onclick="switchGraphTab('entity')" id="tabEntity" class="px-2.5 py-1 rounded bg-blue-600/30 text-blue-300 border border-blue-500/40">Entity (GE)</button>
              <button onclick="switchGraphTab('causal')" id="tabCausal" class="px-2.5 py-1 rounded bg-slate-800 text-slate-400 hover:bg-slate-700">Causal (GC)</button>
              <button onclick="switchGraphTab('flow')" id="tabFlow" class="px-2.5 py-1 rounded bg-slate-800 text-slate-400 hover:bg-slate-700">Flow (GF)</button>
            </div>
          </div>
          <div id="graphContainer" class="network-box"></div>
          <p id="graphDesc" class="text-xs text-slate-400 italic">Showing Entity Graph mapping proprietary assets to Procedure Card.</p>
        </div>
      </div>

      <!-- Right Column: Structure-Aware Generation & Multi-Agent Dispatch -->
      <div class="lg:col-span-6 space-y-6">
        <!-- Generation Quality Card -->
        <div class="glass p-5 rounded-2xl space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Structure-Aware Generation (Verified GF)</h2>
            <div id="sopQualityBadge" class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
              SOP Quality Score: 1.00
            </div>
          </div>
          <div id="generationContent" class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 text-xs font-mono whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto text-slate-200">
            Select a query to generate verified execution steps...
          </div>
          <div id="criteriaScores" class="grid grid-cols-4 gap-2 text-center text-[11px] pt-1 border-t border-slate-800">
            <div class="p-2 bg-slate-900 rounded"><span class="text-slate-400">Step-by-step:</span> <b id="critSteps" class="text-emerald-400">1</b></div>
            <div class="p-2 bg-slate-900 rounded"><span class="text-slate-400">Precise:</span> <b id="critPrec" class="text-emerald-400">1</b></div>
            <div class="p-2 bg-slate-900 rounded"><span class="text-slate-400">Concise:</span> <b id="critConc" class="text-emerald-400">1</b></div>
            <div class="p-2 bg-slate-900 rounded"><span class="text-slate-400">Clear:</span> <b id="critClear" class="text-emerald-400">1</b></div>
          </div>
        </div>

        <!-- Human-in-the-Loop Agent Workflow (Figure 7) -->
        <div class="glass p-5 rounded-2xl space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Agent-Integrated Execution (Figure 7)</h2>
            <span class="text-xs text-amber-400 bg-amber-400/10 px-2.5 py-0.5 rounded border border-amber-400/20">Human Authorization Required</span>
          </div>
          <div id="agentWorkflowList" class="space-y-2 max-h-64 overflow-y-auto">
            <!-- Dynamically populated agent tools -->
          </div>
        </div>
      </div>
    </div>

    <!-- Paper Replication Benchmark Visuals (Figures 4, 5, 6) -->
    <div class="glass p-5 rounded-2xl space-y-4">
      <h2 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Paper Benchmark Replication Figures</h2>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 text-center space-y-2">
          <h3 class="text-xs font-semibold text-slate-300">Figure 4: Retrieval vs Generation Trade-off</h3>
          <img src="/plots/fig4_retrieval_vs_generation.png" class="rounded-lg max-h-48 mx-auto" />
        </div>
        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 text-center space-y-2">
          <h3 class="text-xs font-semibold text-slate-300">Figure 5: Gating Weights KDE</h3>
          <img src="/plots/fig5_gating_weights_kde.png" class="rounded-lg max-h-48 mx-auto" />
        </div>
        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 text-center space-y-2">
          <h3 class="text-xs font-semibold text-slate-300">Figure 6: Retrieval Latency Comparison</h3>
          <img src="/plots/fig6_retrieval_latency.png" class="rounded-lg max-h-48 mx-auto" />
        </div>
      </div>
    </div>
  </main>

  <script>
    let activeSopId = "";
    let activeGraphTab = "entity";
    let graphDataCache = {};
    let network = null;

    function setQuery(text) {
      document.getElementById("queryInput").value = text;
      performQuery();
    }

    async function performQuery() {
      const q = document.getElementById("queryInput").value;
      if (!q) return;

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 4, ablation_mode: "none" })
      });
      const data = await res.json();

      // Update Router Weights
      if (data.router) {
        document.getElementById("weVal").innerText = data.router.w_E.toFixed(3);
        document.getElementById("wcVal").innerText = data.router.w_C.toFixed(3);
        document.getElementById("wfVal").innerText = data.router.w_F.toFixed(3);

        document.getElementById("weBar").style.width = (data.router.w_E * 100) + "%";
        document.getElementById("wcBar").style.width = (data.router.w_C * 100) + "%";
        document.getElementById("wfBar").style.width = (data.router.w_F * 100) + "%";

        document.getElementById("routerIntentSummary").innerHTML =
          `<span><b>Detected Intent:</b> ${data.router.intent_summary}</span>
           <span><b>Extracted Entities:</b> ${data.router.entities.length ? data.router.entities.join(", ") : "None"}</span>`;
      }
      document.getElementById("latencyBadge").innerText = `Latency: ${data.latency_ms} ms`;

      // Update Candidates
      const cList = document.getElementById("candidatesList");
      cList.innerHTML = "";
      if (data.retrieval && data.retrieval.length > 0) {
        activeSopId = data.retrieval[0].sop_id;
        data.retrieval.forEach((item, idx) => {
          const card = document.createElement("div");
          card.className = `p-3 rounded-xl border cursor-pointer transition flex items-center justify-between ${idx === 0 ? "bg-cyan-950/40 border-cyan-500/50" : "bg-slate-900/60 border-slate-800 hover:border-slate-700"}`;
          card.onclick = () => selectSop(item.sop_id);
          card.innerHTML = `
            <div>
              <div class="flex items-center gap-2">
                <span class="text-xs px-2 py-0.5 rounded font-mono font-bold ${idx === 0 ? "bg-cyan-500 text-slate-950" : "bg-slate-800 text-slate-300"}">#${idx+1}</span>
                <span class="text-xs font-semibold text-slate-200">${item.title}</span>
              </div>
              <div class="text-[11px] text-slate-400 mt-1 flex gap-3">
                <span>ID: <code class="text-slate-300">${item.sop_id}</code></span>
                <span>Anchor: <b>${item.anchor_score.toFixed(2)}</b></span>
                <span>Entity: <b>${item.entity_score.toFixed(2)}</b></span>
                <span>Causal: <b>${item.causal_score.toFixed(2)}</b></span>
                <span>Flow: <b>${item.flow_score.toFixed(2)}</b></span>
              </div>
            </div>
            <div class="text-right">
              <span class="text-sm font-bold text-cyan-400 font-mono">${item.final_score.toFixed(4)}</span>
              <span class="block text-[10px] text-slate-500 uppercase">Score R(Q,S)</span>
            </div>
          `;
          cList.appendChild(card);
        });
      }

      // Update Generation
      if (data.generation) {
        document.getElementById("generationContent").innerText = data.generation.response;
        const qScore = data.generation.sop_quality_score;
        document.getElementById("sopQualityBadge").innerText = `SOP Quality Score: ${qScore.score.toFixed(2)}`;
        document.getElementById("critSteps").innerText = qScore.has_step_by_step;
        document.getElementById("critPrec").innerText = qScore.is_precise;
        document.getElementById("critConc").innerText = qScore.is_concise;
        document.getElementById("critClear").innerText = qScore.is_clear;
      }

      // Update Agent Workflow
      const wList = document.getElementById("agentWorkflowList");
      wList.innerHTML = "";
      if (data.workflow && data.workflow.length > 0) {
        data.workflow.forEach(item => {
          const spec = item.agent_spec;
          const div = document.createElement("div");
          div.className = "p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between text-xs";
          div.innerHTML = `
            <div class="space-y-0.5">
              <div class="flex items-center gap-2">
                <span class="font-bold text-slate-200">Step ${item.step_order}</span>
                <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${spec.agent_type === 'Call Agent' ? 'bg-purple-500/20 text-purple-300' : (spec.agent_type === 'Email Agent' ? 'bg-blue-500/20 text-blue-300' : 'bg-emerald-500/20 text-emerald-300')}">${spec.agent_type}</span>
              </div>
              <p class="text-[11px] text-slate-400">${item.step_text}</p>
              <p class="text-[10px] text-slate-500">Action: ${spec.action} &bull; Target: ${spec.target}</p>
            </div>
            <div>
              ${spec.requires_confirmation ?
                '<button onclick="confirmAction(this)" class="px-3 py-1 bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 border border-amber-500/30 rounded font-semibold text-[11px] transition">Authorize</button>' :
                '<span class="text-slate-500 text-[11px]">Auto Exec</span>'}
            </div>
          `;
          wList.appendChild(div);
        });
      }

      // Load graphs for active SOP
      if (activeSopId) {
        loadGraphs(activeSopId);
      }
    }

    function confirmAction(btn) {
      btn.innerText = "Authorized ✓";
      btn.className = "px-3 py-1 bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 rounded font-semibold text-[11px]";
      btn.disabled = true;
    }

    async function selectSop(sopId) {
      activeSopId = sopId;
      await loadGraphs(sopId);
    }

    async function loadGraphs(sopId) {
      const res = await fetch(`/api/sop/${sopId}/graphs`);
      const data = await res.json();
      graphDataCache = data;
      renderCurrentGraph();
    }

    function switchGraphTab(tab) {
      activeGraphTab = tab;
      ["entity", "causal", "flow"].forEach(t => {
        const btn = document.getElementById("tab" + t.charAt(0).toUpperCase() + t.slice(1));
        if (t === tab) {
          btn.className = "px-2.5 py-1 rounded bg-blue-600/30 text-blue-300 border border-blue-500/40 font-bold";
        } else {
          btn.className = "px-2.5 py-1 rounded bg-slate-800 text-slate-400 hover:bg-slate-700";
        }
      });
      renderCurrentGraph();
    }

    function renderCurrentGraph() {
      const container = document.getElementById("graphContainer");
      if (!graphDataCache || !graphDataCache[activeGraphTab]) return;

      const sub = graphDataCache[activeGraphTab];
      const nodes = new vis.DataSet(sub.nodes.map(n => ({
        id: n.id,
        label: n.label,
        color: n.group === 'sop' ? '#06b6d4' : (n.group === 'condition' ? '#f59e0b' : (n.group === 'state' ? '#10b981' : '#3b82f6')),
        font: { color: '#ffffff', size: 11 },
        shape: 'box',
        margin: 8
      })));

      const edges = new vis.DataSet(sub.edges.map(e => ({
        from: e.from,
        to: e.to,
        label: e.label || '',
        arrows: 'to',
        color: { color: '#64748b' },
        font: { color: '#94a3b8', size: 9 }
      })));

      const options = {
        physics: { stabilization: true, barnesHut: { springLength: 90 } },
        interaction: { hover: true, zoomView: true }
      };

      if (network) network.destroy();
      network = new vis.Network(container, { nodes, edges }, options);

      const desc = document.getElementById("graphDesc");
      if (activeGraphTab === "entity") {
        desc.innerText = "Entity Graph (GE): Resolves proprietary structure C1 by linking assets, alarms & parameters.";
      } else if (activeGraphTab === "causal") {
        desc.innerText = "Causal Graph (GC): Resolves condition-dependency C2 by tracing multi-hop symptoms to root cause.";
      } else {
        desc.innerText = "Flow Graph (GF): Resolves actionable response C3 by preserving topological step integrity.";
      }
    }

    // Auto-run on load
    window.onload = () => { performQuery(); };
  </script>
</body>
</html>
"""
