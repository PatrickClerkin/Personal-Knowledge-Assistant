"""
Command-line interface for the Personal Knowledge Assistant.
"""

import click
from pathlib import Path
from typing import Optional

from .ingestion.knowledge_base import KnowledgeBase
from .utils.logger import get_logger

logger = get_logger(__name__)


def _create_kb(index_path: str, strategy: str, chunk_size: int,
               chunk_overlap: int, model: str) -> KnowledgeBase:
    return KnowledgeBase(
        index_path=index_path,
        chunk_strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=model,
    )


@click.group()
@click.option("--index", "-i", default="data/index/default",
              help="Path to the FAISS index (default: data/index/default).", show_default=True)
@click.option("--strategy", "-s", default="sentence",
              type=click.Choice([
                  "fixed", "sentence", "embedding_similarity",
                  "density_clustering", "topic_modeling",
                  "recursive_hierarchical", "auto",
              ]),
              help="Chunking strategy to use.", show_default=True)
@click.option("--chunk-size", "-c", default=512, type=int,
              help="Target chunk size in characters.", show_default=True)
@click.option("--chunk-overlap", default=50, type=int,
              help="Overlap between chunks in characters.", show_default=True)
@click.option("--model", "-m", default="all-MiniLM-L6-v2",
              help="Sentence-transformer model for embeddings.", show_default=True)
@click.pass_context
def cli(ctx, index, strategy, chunk_size, chunk_overlap, model):
    """Personal Knowledge Assistant — document ingestion and semantic search."""
    ctx.ensure_object(dict)
    ctx.obj["index"] = index
    ctx.obj["strategy"] = strategy
    ctx.obj["chunk_size"] = chunk_size
    ctx.obj["chunk_overlap"] = chunk_overlap
    ctx.obj["model"] = model


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=True,
              help="Recursively scan directories.")
@click.pass_context
def ingest(ctx, path, recursive):
    """Ingest a document or directory of documents."""
    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )
    path = Path(path)

    if path.is_file():
        click.echo(f"Ingesting: {path.name}")
        try:
            n_chunks = kb.ingest(path)
            if n_chunks == 0:
                click.echo(click.style("  ~ Skipped (file unchanged since last ingest)", fg="yellow"))
            else:
                click.echo(click.style(f"  ✓ Created {n_chunks} chunks", fg="green"))
        except ValueError as e:
            click.echo(click.style(f"  ✗ {e}", fg="red"))
            raise SystemExit(1)
    elif path.is_dir():
        click.echo(f"Scanning directory: {path}")
        click.echo(f"Supported formats: {', '.join(kb.supported_formats)}")
        stats = kb.ingest_directory(path)
        click.echo()
        click.echo(click.style("Ingestion complete:", bold=True))
        click.echo(f"  Files processed: {stats['files_processed']}")
        click.echo(f"  Files skipped:   {stats['files_skipped']}")
        click.echo(f"  Total chunks:    {stats['total_chunks']}")
        if stats["files_failed"] > 0:
            click.echo(click.style(f"  Files failed:    {stats['files_failed']}", fg="red"))
            for err in stats["errors"]:
                click.echo(f"    - {err['file']}: {err['error']}")
    else:
        click.echo(click.style(f"Invalid path: {path}", fg="red"))
        raise SystemExit(1)

    click.echo(f"\nIndex size: {kb.size} chunks ({len(kb.document_ids)} documents)")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int,
              help="Number of results to return.", show_default=True)
@click.option("--doc-id", "-d", default=None,
              help="Filter results to a specific document ID.")
@click.option("--show-content/--no-content", default=True,
              help="Show chunk content in results.")
@click.option("--rerank/--no-rerank", default=False,
              help="Apply cross-encoder reranking for better precision.")
@click.option("--expand", "-e", default=None,
              type=click.Choice(["synonym", "multi_query", "hyde"]),
              help="Query expansion strategy.")
@click.option("--hybrid/--no-hybrid", default=False,
              help="Use BM25+FAISS hybrid search (Reciprocal Rank Fusion).")
@click.option("--entity-filter", default=None,
              help="Only return chunks containing this entity type e.g. PERSON, ORG, GPE.")
@click.pass_context
def search(ctx, query, top_k, doc_id, show_content, rerank, expand, hybrid, entity_filter):
    """Search the knowledge base with a natural language query.

    \b
    Examples:
        python -m src.cli search "What is dependency injection?"
        python -m src.cli search "design patterns" --top-k 10
        python -m src.cli search "FAISS cosine similarity" --hybrid
        python -m src.cli search "who founded Anthropic" --entity-filter PERSON
    """
    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )

    if kb.size == 0:
        click.echo(click.style(
            "Index is empty. Ingest some documents first: "
            "python -m src.cli ingest <path>", fg="yellow"
        ))
        raise SystemExit(1)

    mode_parts = []
    if hybrid:
        mode_parts.append("hybrid BM25+FAISS")
    if rerank:
        mode_parts.append("reranking")
    if expand:
        mode_parts.append(f"{expand} expansion")
    if entity_filter:
        mode_parts.append(f"entity filter: {entity_filter}")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
    click.echo(f'Searching for: "{query}" (top {top_k}){mode_str}\n')

    if hybrid or rerank or expand or entity_filter:
        results = kb.advanced_search(
            query, top_k=top_k, rerank=rerank,
            expand_query=expand, filter_doc_id=doc_id,
            hybrid=hybrid, label_filter=entity_filter,
        )
    else:
        results = kb.search(query, top_k=top_k, filter_doc_id=doc_id)

    if not results:
        click.echo(click.style("No results found.", fg="yellow"))
        return

    for result in results:
        chunk = result.chunk
        score_colour = "green" if result.score > 0.5 else "yellow" if result.score > 0.3 else "red"

        click.echo(click.style(
            f"[{result.rank}] Score: {result.score:.4f}", fg=score_colour, bold=True
        ))
        click.echo(f"    Source: {chunk.source_doc_title}")

        # Show document version info from the registry
        doc_info = kb.get_document_info(chunk.doc_id)
        if doc_info:
            click.echo(click.style(
                f"    Last updated: {doc_info.updated_at[:10]}", dim=True
            ))

        if chunk.page_number:
            click.echo(f"    Page: {chunk.page_number}")
        click.echo(f"    Chunk: {chunk.chunk_index + 1}/{chunk.total_chunks}")

        # Show entities if present
        entities = chunk.metadata.get("entities", [])
        if entities:
            entity_str = ", ".join(f"{e['text']} ({e['label']})" for e in entities[:5])
            click.echo(f"    Entities: {entity_str}")

        if show_content:
            content = chunk.content.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            click.echo(f"    Content: {content}")
        click.echo()


@cli.command()
@click.pass_context
def info(ctx):
    """Show index statistics and document inventory."""
    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )

    click.echo(click.style("Knowledge Base Info", bold=True))
    click.echo(f"  Index path:    {ctx.obj['index']}")
    click.echo(f"  Strategy:      {ctx.obj['strategy']}")
    click.echo(f"  Chunk size:    {ctx.obj['chunk_size']}")
    click.echo(f"  Chunk overlap: {ctx.obj['chunk_overlap']}")
    click.echo(f"  Model:         {ctx.obj['model']}")
    click.echo(f"  Total chunks:  {kb.size}")
    click.echo(f"  Documents:     {len(kb.document_ids)}")
    click.echo(f"  Supported:     {', '.join(kb.supported_formats)}")

    if kb.document_ids:
        click.echo(click.style("\nDocuments:", bold=True))
        for doc_id in kb.document_ids:
            chunks = kb.get_document_chunks(doc_id)
            if chunks:
                title = chunks[0].source_doc_title
                doc_info = kb.get_document_info(doc_id)
                version_str = ""
                if doc_info:
                    version_str = (
                        f"  ingested {doc_info.ingested_at[:10]}"
                        + (f"  updated {doc_info.updated_at[:10]}"
                           if doc_info.updated_at != doc_info.ingested_at else "")
                    )
                click.echo(
                    f"  [{doc_id[:8]}...] {title} ({len(chunks)} chunks)"
                    + click.style(version_str, dim=True)
                )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--strategies", "-s", default=None,
              help="Comma-separated strategies to compare (default: all simple + embedding_similarity).")
@click.pass_context
def compare(ctx, path, strategies):
    """Compare chunking strategies on a document."""
    from .ingestion.document_manager import DocumentManager
    from .ingestion.chunking.chunk_manager import ChunkManager

    dm = DocumentManager()
    doc = dm.parse_document(Path(path))

    strategy_list = None
    if strategies:
        strategy_list = [s.strip() for s in strategies.split(",")]

    cm = ChunkManager(
        chunk_size=ctx.obj["chunk_size"],
        chunk_overlap=ctx.obj["chunk_overlap"],
    )
    results = cm.compare_strategies(doc, strategies=strategy_list)

    click.echo(click.style(f"Strategy Comparison: {Path(path).name}", bold=True))
    click.echo(f"Document length: {len(doc.content):,} characters\n")

    click.echo(f"{'Strategy':<25} {'Chunks':>7} {'Avg Size':>9} {'Min':>6} {'Max':>6} {'Coverage':>9}")
    click.echo("─" * 67)

    for strategy, data in results.items():
        if "error" in data:
            click.echo(f"{strategy:<25} {click.style('ERROR: ' + data['error'], fg='red')}")
        else:
            coverage_pct = f"{data['coverage_ratio']:.1%}"
            click.echo(
                f"{strategy:<25} {data['n_chunks']:>7} "
                f"{data['avg_size']:>9.0f} {data['min_size']:>6} "
                f"{data['max_size']:>6} {coverage_pct:>9}"
            )


@cli.command("delete")
@click.argument("doc_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete_doc(ctx, doc_id, confirm):
    """Delete a document from the index by its ID."""
    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )

    if not confirm:
        click.confirm(f"Delete document {doc_id}?", abort=True)

    deleted = kb.delete_document(doc_id)
    if deleted > 0:
        click.echo(click.style(f"Deleted {deleted} chunks.", fg="green"))
    else:
        click.echo(click.style(f"No document found with ID: {doc_id}", fg="yellow"))


@cli.command()
@click.option("--confirm", is_flag=True, required=True,
              help="Required flag to confirm clearing the index.")
@click.pass_context
def clear(ctx, confirm):
    """Clear the entire index. Requires --confirm flag."""
    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )
    size_before = kb.size
    kb.clear()
    if kb.index_path:
        kb.save()
    click.echo(click.style(f"Index cleared. Removed {size_before} chunks.", fg="green"))


@cli.command("eval")
@click.argument("judgments_path", type=click.Path(exists=True))
@click.option("--top-k", "-k", default=5, type=int,
              help="Number of results to evaluate.", show_default=True)
@click.pass_context
def evaluate(ctx, judgments_path, top_k):
    """Evaluate retrieval quality against relevance judgments."""
    from .retrieval.evaluation import RetrievalEvaluator

    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )

    if kb.size == 0:
        click.echo(click.style("Index is empty.", fg="yellow"))
        raise SystemExit(1)

    evaluator = RetrievalEvaluator()
    n_loaded = evaluator.load_judgments(Path(judgments_path))
    click.echo(f"Loaded {n_loaded} relevance judgments")
    click.echo(f"Evaluating at K={top_k}...\n")

    eval_results = evaluator.evaluate(kb, top_k=top_k)
    agg = eval_results["aggregate"]

    click.echo(click.style("Aggregate Metrics:", bold=True))
    click.echo(f"  Queries evaluated: {agg['num_queries']}")
    click.echo(f"  Precision@{top_k}:    {agg['mean_precision_at_k']:.4f}")
    click.echo(f"  Recall@{top_k}:       {agg['mean_recall_at_k']:.4f}")
    click.echo(f"  MRR:              {agg['mrr']:.4f}")
    click.echo(f"  nDCG@{top_k}:         {agg['mean_ndcg_at_k']:.4f}")
    click.echo(f"  MAP:              {agg['map']:.4f}")

    click.echo(click.style("\nPer-Query Breakdown:", bold=True))
    for result in eval_results["queries"]:
        rr_colour = "green" if result.reciprocal_rank > 0.5 else "yellow" if result.reciprocal_rank > 0 else "red"
        click.echo(f'  "{result.query[:50]}"')
        click.echo(
            f"    P@{top_k}={result.precision_at_k:.2f}  "
            f"R@{top_k}={result.recall_at_k:.2f}  "
            + click.style(f"RR={result.reciprocal_rank:.2f}", fg=rr_colour)
            + f"  nDCG={result.ndcg_at_k:.2f}"
        )


@cli.command()
@click.option("--top-k", "-k", default=5, type=int,
              help="Number of context chunks per query.", show_default=True)
@click.option("--rerank/--no-rerank", default=False,
              help="Apply cross-encoder reranking.")
@click.option("--expand", "-e", default=None,
              type=click.Choice(["synonym", "multi_query", "hyde"]),
              help="Query expansion strategy.")
@click.option("--hybrid/--no-hybrid", default=False,
              help="Use BM25+FAISS hybrid retrieval in the RAG pipeline.")
@click.pass_context
def chat(ctx, top_k, rerank, expand, hybrid):
    """Interactive RAG-powered chat with the knowledge base."""
    from .rag.llm import ClaudeProvider
    from .rag.pipeline import RAGPipeline

    kb = _create_kb(
        ctx.obj["index"], ctx.obj["strategy"],
        ctx.obj["chunk_size"], ctx.obj["chunk_overlap"], ctx.obj["model"],
    )

    if kb.size == 0:
        click.echo(click.style("Index is empty. Ingest documents first.", fg="yellow"))
        raise SystemExit(1)

    llm = ClaudeProvider()
    if not llm.is_available():
        click.echo(click.style(
            "ANTHROPIC_API_KEY not set. Export it to use the chat feature.", fg="red",
        ))
        raise SystemExit(1)

    rag = RAGPipeline(
        knowledge_base=kb,
        llm_provider=llm,
        top_k=top_k,
        rerank=rerank,
        expand_query=expand,
        hybrid=hybrid,
    )

    mode_parts = []
    if hybrid:
        mode_parts.append("hybrid")
    if rerank:
        mode_parts.append("rerank")
    if expand:
        mode_parts.append(expand)
    mode_str = f" [{', '.join(mode_parts)}]" if mode_parts else ""

    click.echo(click.style(f"Knowledge Assistant Chat{mode_str}", bold=True))
    click.echo(f"Index: {kb.size} chunks from {len(kb.document_ids)} documents")
    click.echo("Type 'quit' to exit, 'clear' to reset conversation.\n")

    while True:
        try:
            question = click.prompt(
                click.style("You", fg="cyan", bold=True),
                prompt_suffix=": ",
            )
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye!")
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            click.echo("Goodbye!")
            break
        if question.lower() == "clear":
            rag.clear_history()
            click.echo(click.style("Conversation cleared.\n", fg="yellow"))
            continue

        try:
            response = rag.query(question)
            click.echo()
            click.echo(click.style("Assistant", fg="green", bold=True) + ": " + response.answer)
            click.echo()

            if response.sources:
                click.echo(click.style("Sources:", dim=True))
                seen_sources = set()
                for r in response.sources[:3]:
                    source = r.chunk.source_doc_title
                    page = f" p.{r.chunk.page_number}" if r.chunk.page_number else ""
                    key = f"{source}{page}"
                    if key not in seen_sources:
                        seen_sources.add(key)
                        click.echo(click.style(
                            f"  • {source}{page} (score: {r.score:.3f})", dim=True
                        ))
                click.echo()

        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))
            click.echo()


if __name__ == "__main__":
    cli()