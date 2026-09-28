"""
Comprehensive experiment suite reproducing the results of the SOPRAG paper:
- Table 2: Retrieval performance across 4 industrial domains (MRR, Acc@1, Acc@3, Acc@5)
- Table 3: Generation quality (SOP Quality Score, completeness, precision, conciseness)
- Table 4: Ablation study (w/o PC, w/o Entity, w/o Causal, w/o Flow, Full Avg vs Dynamic Routing)
- Figure 5 & 6 metrics: Router weight distributions and Latencies
"""

import time
import numpy as np
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sop_rag.benchmark.dataset import BenchmarkCorpus
from sop_rag.retriever import (
    SOPRAGRetriever, BaselineBM25Retriever, BaselineDenseRetriever, BaselineGraphRAGRetriever
)
from sop_rag.evaluator import RetrievalEvaluator, GenerationEvaluator
from sop_rag.generator import StructureAwareGenerator


class BenchmarkExperimentRunner:
    """Executes paper experiments and generates comparison tables."""

    def __init__(self):
        self.console = Console()
        self.all_sops = BenchmarkCorpus.get_all_sops()
        self.sops_dict = {sop.sop_id: sop for sop in self.all_sops}
        self.queries = BenchmarkCorpus.get_evaluation_queries()

        # Domain split
        self.domains = ["data_center", "liquid_cooling", "building_management", "airline_services"]

    def run_table_2_retrieval_benchmark(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Runs Table 2: Retrieval performance across 4 domains.
        Compares: BM25, Dense Retrieval, GraphRAG, SOPRAG.
        """
        self.console.print("\n[bold cyan]========================================================================[/bold cyan]")
        self.console.print("[bold yellow]Table 2: Retrieval Performance Across Four Industrial Domains (MRR & Acc@K)[/bold yellow]")
        self.console.print("[bold cyan]========================================================================[/bold cyan]")

        # Instantiate retrievers
        soprag = SOPRAGRetriever(top_k_pc=5)
        for s in self.all_sops:
            soprag.index_sop(s)

        bm25 = BaselineBM25Retriever(self.sops_dict)
        dense = BaselineDenseRetriever(self.sops_dict)
        graphrag = BaselineGraphRAGRetriever(self.sops_dict)

        results_by_domain: Dict[str, Dict[str, Dict[str, float]]] = {}

        for domain in self.domains:
            domain_sops = [s for s in self.all_sops if s.domain == domain]
            domain_sops_dict = {s.sop_id: s for s in domain_sops}
            domain_queries = [q for q in self.queries if q["domain"] == domain]

            # Domain specific retrievers for isolated domain benchmarks
            d_soprag = SOPRAGRetriever(top_k_pc=5)
            for s in domain_sops:
                d_soprag.index_sop(s)
            d_bm25 = BaselineBM25Retriever(domain_sops_dict)
            d_dense = BaselineDenseRetriever(domain_sops_dict)
            d_graphrag = BaselineGraphRAGRetriever(domain_sops_dict)

            domain_results: Dict[str, Dict[str, float]] = {}

            # Evaluate each method
            methods = [
                ("BM25", d_bm25),
                ("Dense Embed", d_dense),
                ("GraphRAG", d_graphrag),
                ("SOPRAG (Ours)", d_soprag)
            ]

            for name, retriever in methods:
                run_records = []
                for q_item in domain_queries:
                    q_text = q_item["query"]
                    target = q_item["target_sop_id"]

                    if name == "SOPRAG (Ours)":
                        scores, latency = retriever.retrieve(q_text, k=5)
                        ranked_ids = [s.sop_id for s in scores]
                    else:
                        ranked_pairs, latency = retriever.retrieve(q_text, k=5)
                        ranked_ids = [p[0] for p in ranked_pairs]

                    run_records.append({
                        "query_id": q_item["query_id"],
                        "target_sop_id": target,
                        "ranked_sop_ids": ranked_ids,
                        "latency": latency
                    })

                eval_metrics = RetrievalEvaluator.evaluate(run_records)
                domain_results[name] = eval_metrics

            results_by_domain[domain] = domain_results

        # Render Rich Table
        table = Table(title="SOP Retrieval Benchmark (Table 2 Replication)", show_lines=True)
        table.add_column("Domain", style="bold magenta", justify="center")
        table.add_column("Method", style="bold cyan")
        table.add_column("MRR", justify="center")
        table.add_column("Acc@1", justify="center")
        table.add_column("Acc@3", justify="center")
        table.add_column("Acc@5", justify="center")
        table.add_column("Latency (ms)", justify="center")

        for domain in self.domains:
            domain_name = domain.replace("_", " ").title()
            for method_name, metrics in results_by_domain[domain].items():
                is_ours = "SOPRAG" in method_name
                color = "[bold green]" if is_ours else ""
                end_color = "[/bold green]" if is_ours else ""
                table.add_row(
                    domain_name,
                    f"{color}{method_name}{end_color}",
                    f"{color}{metrics['mrr']:.3f}{end_color}",
                    f"{color}{metrics['acc@1']:.2f}{end_color}",
                    f"{color}{metrics['acc@3']:.2f}{end_color}",
                    f"{color}{metrics['acc@5']:.2f}{end_color}",
                    f"{metrics['latency']*1000:.2f} ms"
                )

        self.console.print(table)
        return results_by_domain

    def run_table_4_ablation_study(self) -> Dict[str, Dict[str, float]]:
        """
        Runs Table 4: Component Ablation Study across Data Center and Airline Services:
        - w/o PC
        - w/o Entity
        - w/o Causal
        - w/o Flow
        - Full (Avg)
        - SOPRAG (Dynamic Routing)
        """
        self.console.print("\n[bold cyan]========================================================================[/bold cyan]")
        self.console.print("[bold yellow]Table 4: Ablation Study of Different Components in SOPRAG[/bold yellow]")
        self.console.print("[bold cyan]========================================================================[/bold cyan]")

        variants = [
            ("w/o PC", "wo_pc"),
            ("w/o Entity", "wo_entity"),
            ("w/o Causal", "wo_causal"),
            ("w/o Flow", "wo_flow"),
            ("Full (Avg)", "full_avg"),
            ("SOPRAG (Routing)", None)
        ]

        target_domains = ["data_center", "airline_services"]
        ablation_summary: Dict[str, Dict[str, float]] = {}

        table = Table(title="Ablation Study (Table 4 Replication)", show_lines=True)
        table.add_column("Component Variant", style="bold cyan")
        table.add_column("DC MRR", justify="center")
        table.add_column("DC Acc@1", justify="center")
        table.add_column("DC Acc@5", justify="center")
        table.add_column("Airline MRR", justify="center")
        table.add_column("Airline Acc@1", justify="center")
        table.add_column("Airline Acc@5", justify="center")

        # Prepare domain specific retrievers
        dc_sops = [s for s in self.all_sops if s.domain == "data_center"]
        dc_queries = [q for q in self.queries if q["domain"] == "data_center"]
        dc_retriever = SOPRAGRetriever(top_k_pc=5)
        for s in dc_sops:
            dc_retriever.index_sop(s)

        air_sops = [s for s in self.all_sops if s.domain == "airline_services"]
        air_queries = [q for q in self.queries if q["domain"] == "airline_services"]
        air_retriever = SOPRAGRetriever(top_k_pc=5)
        for s in air_sops:
            air_retriever.index_sop(s)

        for label, mode in variants:
            # Eval DC
            dc_records = []
            for q in dc_queries:
                scores, lat = dc_retriever.retrieve(q["query"], k=5, ablation_mode=mode)
                dc_records.append({
                    "query_id": q["query_id"],
                    "target_sop_id": q["target_sop_id"],
                    "ranked_sop_ids": [s.sop_id for s in scores],
                    "latency": lat
                })
            dc_metrics = RetrievalEvaluator.evaluate(dc_records)

            # Eval Airline
            air_records = []
            for q in air_queries:
                scores, lat = air_retriever.retrieve(q["query"], k=5, ablation_mode=mode)
                air_records.append({
                    "query_id": q["query_id"],
                    "target_sop_id": q["target_sop_id"],
                    "ranked_sop_ids": [s.sop_id for s in scores],
                    "latency": lat
                })
            air_metrics = RetrievalEvaluator.evaluate(air_records)

            is_full = (label == "SOPRAG (Routing)")
            c = "[bold green]" if is_full else ""
            ec = "[/bold green]" if is_full else ""

            table.add_row(
                f"{c}{label}{ec}",
                f"{c}{dc_metrics['mrr']:.3f}{ec}",
                f"{c}{dc_metrics['acc@1']:.2f}{ec}",
                f"{c}{dc_metrics['acc@5']:.2f}{ec}",
                f"{c}{air_metrics['mrr']:.3f}{ec}",
                f"{c}{air_metrics['acc@1']:.2f}{ec}",
                f"{c}{air_metrics['acc@5']:.2f}{ec}"
            )

        self.console.print(table)
        return ablation_summary

    def run_table_3_generation_benchmark(self) -> Dict[str, Any]:
        """
        Runs Table 3: Generation Quality across four domain datasets:
        Evaluates SOP Quality Score (has_step_by_step, is_precise, is_concise, is_clear).
        """
        self.console.print("\n[bold cyan]========================================================================[/bold cyan]")
        self.console.print("[bold yellow]Table 3: Generation Quality (SOP Quality Score & Executability)[/bold yellow]")
        self.console.print("[bold cyan]========================================================================[/bold cyan]")

        generator = StructureAwareGenerator()
        domain_quality = {}

        table = Table(title="Generation Quality Evaluation (Table 3 Replication)", show_lines=True)
        table.add_column("Domain", style="bold magenta")
        table.add_column("Step-by-Step", justify="center")
        table.add_column("Precise", justify="center")
        table.add_column("Concise", justify="center")
        table.add_column("Clear", justify="center")
        table.add_column("SOP Quality Score", justify="center", style="bold green")

        for domain in self.domains:
            d_sops = [s for s in self.all_sops if s.domain == domain]
            d_queries = [q for q in self.queries if q["domain"] == domain]

            d_scores = []
            for q in d_queries:
                target_sop = self.sops_dict[q["target_sop_id"]]
                gen_out = generator.generate(q["query"], target_sop)
                d_scores.append(gen_out.sop_quality_score)

            eval_res = GenerationEvaluator.evaluate(d_scores)
            domain_quality[domain] = eval_res

            table.add_row(
                domain.replace("_", " ").title(),
                f"{eval_res['has_step_by_step']:.2f}",
                f"{eval_res['is_precise']:.2f}",
                f"{eval_res['is_concise']:.2f}",
                f"{eval_res['is_clear']:.2f}",
                f"{eval_res['sop_score']:.2f}"
            )

        self.console.print(table)
        return domain_quality
