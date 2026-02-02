// Doc-MLIR-GraphDB pipeline reference MLIR.
//
// This file exists primarily as a stable anchor target for documentation
// footnotes in:
// - docs/sdlc/01_tt_metal/02_Architecture/00_Doc_MLIR_GraphDB_Pipeline.md
//
// NOTE: The concrete ops/types are intentionally minimal for now; the contract
// is that the anchors below exist and remain stable.

module attributes {
  // dialect = ttm.sdlc_doc
} {
  // #intro
  // Concept: Documentation pipeline overview.

  // #intro_chain
  // Concept: Doc <-> MLIR <-> GraphDB chain, LLM only at the end.

  // #doc_layer
  // Concept: Markdown document layer.

  // #mlir_doc_layer
  // Concept: MLIR document-as-structure layer (ttm.sdlc_doc).

  // #mlir_passes
  // Concept: MLIR transformation chain (passes/pipelines).

  // #graphdb
  // Concept: GraphDB (KG) representation and SPARQL queries.

  // #next_mlir
  // Concept: Next MLIR layer materialized from GraphDB/query results.

  // #final_llm
  // Concept: Final LLM stage: MLIR -> technical text.
}

