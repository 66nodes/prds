// AI Agent Platform - GraphRAG Schema for Neo4j
// Creates the graph structure for Hybrid RAG knowledge representation
// Compatible with Microsoft GraphRAG framework

// ================================
// CORE ENTITY TYPES
// ================================

// Documents and Text Sources
CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT text_chunk_id_unique IF NOT EXISTS FOR (t:TextChunk) REQUIRE t.id IS UNIQUE;

// Knowledge Graph Entities
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (cl:Claim) REQUIRE cl.id IS UNIQUE;

// Agents and Processing
CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE;

// ================================
// DOCUMENT MANAGEMENT NODES
// ================================

// Document: Represents source documents
// Properties: id, title, content, source_type, created_at, updated_at, metadata
CREATE INDEX document_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title);
CREATE INDEX document_source_type_idx IF NOT EXISTS FOR (d:Document) ON (d.source_type);
CREATE INDEX document_created_idx IF NOT EXISTS FOR (d:Document) ON (d.created_at);

// TextChunk: Segmented portions of documents for processing
// Properties: id, content, chunk_index, token_count, embedding_vector_id, created_at
CREATE INDEX textchunk_chunk_index_idx IF NOT EXISTS FOR (t:TextChunk) ON (t.chunk_index);
CREATE INDEX textchunk_embedding_idx IF NOT EXISTS FOR (t:TextChunk) ON (t.embedding_vector_id);

// ================================
// KNOWLEDGE GRAPH NODES
// ================================

// Entity: Named entities extracted from text (people, places, concepts, etc.)
// Properties: id, name, type, description, importance_score, frequency, created_at
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_importance_idx IF NOT EXISTS FOR (e:Entity) ON (e.importance_score);

// Community: Hierarchical clusters of related entities
// Properties: id, name, level, description, summary, importance_score, entity_count
CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level);
CREATE INDEX community_importance_idx IF NOT EXISTS FOR (c:Community) ON (c.importance_score);

// Claim: Factual statements extracted from text
// Properties: id, text, confidence_score, source_refs, created_at, validated
CREATE INDEX claim_confidence_idx IF NOT EXISTS FOR (cl:Claim) ON (cl.confidence_score);
CREATE INDEX claim_validated_idx IF NOT EXISTS FOR (cl:Claim) ON (cl.validated);

// ================================
// AI AGENT NODES
// ================================

// Agent: AI agents that process and generate content
// Properties: id, name, type, model, version, capabilities, created_at
CREATE INDEX agent_type_idx IF NOT EXISTS FOR (a:Agent) ON (a.type);
CREATE INDEX agent_model_idx IF NOT EXISTS FOR (a:Agent) ON (a.model);

// Session: Processing sessions with context and state
// Properties: id, agent_id, status, started_at, ended_at, context, metadata
CREATE INDEX session_status_idx IF NOT EXISTS FOR (s:Session) ON (s.status);
CREATE INDEX session_started_idx IF NOT EXISTS FOR (s:Session) ON (s.started_at);

// ================================
// RELATIONSHIP TYPES
// ================================

// Document Relationships
// Document -> TextChunk: HAS_CHUNK (properties: chunk_order)
// TextChunk -> Entity: MENTIONS (properties: frequency, context)
// TextChunk -> Claim: SUPPORTS (properties: confidence, evidence_strength)

// Knowledge Graph Relationships
// Entity -> Entity: RELATED_TO (properties: relationship_type, strength, context)
// Entity -> Community: BELONGS_TO (properties: membership_strength, role)
// Community -> Community: CONTAINS (properties: hierarchy_level)
// Claim -> Entity: ABOUT (properties: relevance_score)
// Claim -> Claim: CONTRADICTS | SUPPORTS (properties: confidence, reasoning)

// Agent Relationships
// Agent -> Session: CREATED (properties: created_at)
// Session -> Document: PROCESSED (properties: processing_time, status)
// Session -> Entity: EXTRACTED (properties: extraction_confidence)
// Session -> Claim: GENERATED (properties: generation_confidence)

// Vector Embeddings Reference
// TextChunk -> VectorEmbedding: HAS_EMBEDDING (properties: vector_id, model_used)
// Entity -> VectorEmbedding: HAS_EMBEDDING (properties: vector_id, model_used)

// ================================
// VALIDATION SCHEMA
// ================================

// Validation nodes for GraphRAG quality control
CREATE CONSTRAINT validation_id_unique IF NOT EXISTS FOR (v:Validation) REQUIRE v.id IS UNIQUE;

// Validation: Tracks validation of generated content
// Properties: id, target_type, target_id, validation_type, score, details, created_at
CREATE INDEX validation_target_type_idx IF NOT EXISTS FOR (v:Validation) ON (v.target_type);
CREATE INDEX validation_score_idx IF NOT EXISTS FOR (v:Validation) ON (v.score);
CREATE INDEX validation_created_idx IF NOT EXISTS FOR (v:Validation) ON (v.created_at);

// ================================
// PERFORMANCE INDEXES
// ================================

// Composite indexes for common query patterns
CREATE INDEX entity_type_importance_idx IF NOT EXISTS FOR (e:Entity) ON (e.type, e.importance_score);
CREATE INDEX community_level_importance_idx IF NOT EXISTS FOR (c:Community) ON (c.level, e.importance_score);
CREATE INDEX claim_confidence_validated_idx IF NOT EXISTS FOR (cl:Claim) ON (cl.confidence_score, cl.validated);

// Full-text search indexes
CREATE FULLTEXT INDEX entity_search_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description];
CREATE FULLTEXT INDEX document_search_idx IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content];
CREATE FULLTEXT INDEX claim_search_idx IF NOT EXISTS FOR (cl:Claim) ON EACH [cl.text];

// ================================
// SAMPLE DATA STRUCTURE
// ================================

// Example document node
// CREATE (d:Document {
//   id: "doc_001",
//   title: "AI Strategy Document",
//   source_type: "pdf",
//   created_at: datetime(),
//   metadata: {
//     author: "Strategy Team",
//     department: "AI Research",
//     classification: "internal"
//   }
// });

// Example entity node
// CREATE (e:Entity {
//   id: "entity_001",
//   name: "Machine Learning",
//   type: "CONCEPT",
//   description: "A method of data analysis that automates analytical model building",
//   importance_score: 0.85,
//   frequency: 42,
//   created_at: datetime()
// });

// Example community node
// CREATE (c:Community {
//   id: "community_001",
//   name: "AI Technologies",
//   level: 1,
//   description: "Community of entities related to artificial intelligence technologies",
//   importance_score: 0.92,
//   entity_count: 15
// });

// ================================
// QUERY OPTIMIZATION HINTS
// ================================

// For GraphRAG validation queries:
// - Use entity importance scores for filtering
// - Leverage community hierarchies for context
// - Cross-reference claims with multiple entities
// - Track validation scores for quality metrics

// For retrieval augmentation:
// - Start with high-importance entities
// - Follow relationship paths with strong connections
// - Use community summaries for broader context
// - Prioritize validated claims and entities

// ================================
// MONITORING QUERIES
// ================================

// Entity count by type
// MATCH (e:Entity) RETURN e.type, count(e) ORDER BY count(e) DESC;

// Community hierarchy depth
// MATCH (c:Community) RETURN c.level, count(c) ORDER BY c.level;

// Validation coverage
// MATCH (v:Validation) RETURN v.target_type, avg(v.score), count(v);

// Recent processing activity
// MATCH (s:Session) WHERE s.started_at > datetime() - duration('P1D')
// RETURN s.status, count(s);