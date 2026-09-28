"""
Main entry point for SOPRAG.
Supports:
1. run-experiments: Reproduce paper tables (Table 2, 3, 4) and figures (Fig 4, 5, 6).
2. query: Query SOPRAG directly with step-by-step verified execution and router analysis.
3. demo: Interactive demonstration across the 4 industrial domains.
4. serve: Start local FastAPI / Generative UI web dashboard.
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sop_rag.benchmark.dataset import BenchmarkCorpus
from sop_rag.pipeline import SOPRAGPipeline
from sop_rag.experiments.run_benchmarks import BenchmarkExperimentRunner
from sop_rag.experiments.visualization import plot_paper_figures


console = Console()


def run_experiments():
    console.print("\n[bold green]Starting SOPRAG Paper Replication Experiments...[/bold green]")
    runner = BenchmarkExperimentRunner()

    # Table 2: Retrieval performance
    runner.run_table_2_retrieval_benchmark()

    # Table 4: Ablation study
    runner.run_table_4_ablation_study()

    # Table 3: Generation Quality
    runner.run_table_3_generation_benchmark()

    # Figures 4, 5, 6
    console.print("\n[bold cyan]Generating paper visualization figures...[/bold cyan]")
    paths = plot_paper_figures("benchmark_results")
    console.print(f"[bold green]✓ Plots saved to:[/bold green]")
    for name, p in paths.items():
        console.print(f"  - {name}: [link={p}]{p}[/link]")


def run_query(query_text: str):
    console.print(Panel(f"[bold white]{query_text}[/bold white]", title="[bold yellow]User Operator Query[/bold yellow]"))

    pipeline = SOPRAGPipeline()
    sops = BenchmarkCorpus.get_all_sops()
    pipeline.load_documents(sops)

    res = pipeline.query(query_text, top_k=3, generate_response=True, plan_agents=True)

    # 1. Router Analysis
    rw = res["router_weights"]
    if rw:
        router_table = Table(title="LLM Router Dynamic Attention Allocation w = [w_E, w_C, w_F]", show_lines=True)
        router_table.add_column("Expert Graph", style="bold")
        router_table.add_column("Weight", justify="center")
        router_table.add_column("Target Reasoning Focus", style="dim")

        router_table.add_row("Entity Graph (w_E)", f"{rw.w_E:.3f}", "Proprietary alarms, parameters, asset models")
        router_table.add_row("Causal Graph (w_C)", f"{rw.w_C:.3f}", "Root causes, symptoms, diagnostic 'why'")
        router_table.add_row("Flow Graph (w_F)", f"{rw.w_F:.3f}", "Step-by-step procedures, execution 'how'")
        console.print(router_table)
        console.print(f"[bold cyan]Intent Summary:[/bold cyan] {rw.intent_summary}")
        if rw.extracted_entities:
            console.print(f"[bold cyan]Extracted Proprietary Entities:[/bold cyan] {', '.join(rw.extracted_entities)}")

    # 2. Retrieval Ranking Breakdown
    console.print(f"\n[bold green]Retrieval Latency:[/bold green] {res['retrieval_latency_ms']} ms")
    rank_table = Table(title="Top Retrieved Candidate SOPs", show_lines=True)
    rank_table.add_column("Rank", justify="center")
    rank_table.add_column("SOP ID", style="bold cyan")
    rank_table.add_column("SOP Title")
    rank_table.add_column("Final Score R(Q,S)", justify="center", style="bold green")
    rank_table.add_column("Anchor Sim", justify="center")
    rank_table.add_column("Entity (R_E)", justify="center")
    rank_table.add_column("Causal (R_C)", justify="center")
    rank_table.add_column("Flow (R_F)", justify="center")

    for idx, r in enumerate(res["retrieval_results"]):
        rank_table.add_row(
            str(idx + 1),
            r.sop_id,
            r.title,
            f"{r.final_score:.4f}",
            f"{r.anchor_score:.3f}",
            f"{r.entity_score:.3f}",
            f"{r.causal_score:.3f}",
            f"{r.flow_score:.3f}"
        )
    console.print(rank_table)

    # 3. Structure-Aware Generation
    gen = res["generation"]
    if gen:
        console.print(Panel(gen.generated_response, title="[bold green]Structure-Aware Generation (Verified Flow GF)[/bold green]"))
        q_score = gen.sop_quality_score
        console.print(f"[bold yellow]SOP Quality Score:[/bold yellow] [bold]{q_score['score']:.2f}[/bold] (Step-by-step: {q_score['has_step_by_step']}, Precise: {q_score['is_precise']}, Concise: {q_score['is_concise']}, Clear: {q_score['is_clear']})")

    # 4. Agent-Integrated Operational Execution (Figure 7)
    workflow = res["agent_workflow"]
    if workflow:
        agent_table = Table(title="Agent-Integrated Operational Execution (Human-in-the-Loop Workflow)", show_lines=True)
        agent_table.add_column("Step", justify="center", style="bold")
        agent_table.add_column("Assigned Agent / Tool", style="bold magenta")
        agent_table.add_column("Action Type")
        agent_table.add_column("Target / Dispatch")
        agent_table.add_column("Human Auth Required?", justify="center")

        for item in workflow:
            spec = item["agent_spec"]
            req_auth = "[bold yellow]YES (Operator)[/bold yellow]" if spec["requires_confirmation"] else "[green]Auto / Field[/green]"
            agent_table.add_row(
                f"Step {item['step_order']}",
                spec["agent_type"],
                spec["action"],
                spec["target"],
                req_auth
            )
        console.print(agent_table)


def run_interactive_demo():
    console.print("[bold cyan]========================================================================[/bold cyan]")
    console.print("[bold green]Welcome to SOPRAG: Industrial Standard Operating Procedures RAG[/bold green]")
    console.print("[bold cyan]========================================================================[/bold cyan]")

    sample_queries = [
        "How can I view and monitor the real-time operating status of HVAC equipment in the chilled water and cooling water systems?",
        "Why is CDU pump cavitation causing vibration and what causes entrapped air in liquid cooling?",
        "What is the standard procedure for recovering from an aircraft upset or unusual attitude in flight?",
        "How to initiate a hard server reboot using iDRAC or ILO console?",
        "Which ball valves V-201 and V-202 must be isolated during ALARM-LEAK-101?"
    ]

    console.print("\nSample queries from the paper to try:")
    for i, sq in enumerate(sample_queries):
        console.print(f"[{i+1}] {sq}")

    console.print("\nRunning demonstration query #1...")
    run_query(sample_queries[3])

    console.print("\nRunning demonstration query #2 (Causal diagnosis)...")
    run_query(sample_queries[1])

    console.print("\nRunning demonstration query #3 (Flow recovery checklist)...")
    run_query(sample_queries[2])


def main():
    parser = argparse.ArgumentParser(description="SOPRAG CLI and Benchmark Runner")
    parser.add_argument("mode", nargs="?", default="run-experiments", choices=["run-experiments", "query", "demo"], help="Execution mode")
    parser.add_argument("--query", "-q", type=str, default="", help="Query string when mode is 'query'")

    args = parser.parse_args()

    if args.mode == "run-experiments":
        run_experiments()
    elif args.mode == "demo":
        run_interactive_demo()
    elif args.mode == "query":
        q = args.query if args.query else "What is the procedure for handling a critical server outage?"
        run_query(q)


if __name__ == "__main__":
    main()
