#!/usr/bin/env python3
"""
AI Agent Platform - Milvus Collection Setup
Creates collections and indexes for Hybrid RAG vector storage
"""

import sys
from pymilvus import (
    connections, db, Collection, CollectionSchema, FieldSchema, 
    DataType, utility, Index
)

def connect_milvus():
    """Connect to Milvus server"""
    try:
        connections.connect(
            alias="default",
            host="localhost",
            port="19530"
        )
        print("✅ Connected to Milvus")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

def create_text_chunks_collection():
    """Create collection for text chunk embeddings"""
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="token_count", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # OpenAI embedding size
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Text chunks with embeddings for GraphRAG retrieval",
        enable_dynamic_field=True
    )
    
    # Create collection
    collection_name = "text_chunks"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️  Dropped existing collection: {collection_name}")
    
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection: {collection_name}")
    
    # Create index for vector similarity search
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",  # Cosine similarity for text embeddings
        "params": {"nlist": 128}
    }
    
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Created vector index for {collection_name}")
    
    # Create scalar indexes for filtering
    collection.create_index(field_name="document_id")
    collection.create_index(field_name="chunk_index")
    print(f"✅ Created scalar indexes for {collection_name}")
    
    return collection

def create_entities_collection():
    """Create collection for entity embeddings"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="importance_score", dtype=DataType.DOUBLE),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Entity embeddings for semantic entity retrieval",
        enable_dynamic_field=True
    )
    
    collection_name = "entities"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️  Dropped existing collection: {collection_name}")
    
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection: {collection_name}")
    
    # Vector index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Created vector index for {collection_name}")
    
    # Scalar indexes
    collection.create_index(field_name="type")
    collection.create_index(field_name="importance_score")
    print(f"✅ Created scalar indexes for {collection_name}")
    
    return collection

def create_communities_collection():
    """Create collection for community embeddings"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="level", dtype=DataType.INT64),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="importance_score", dtype=DataType.DOUBLE),
        FieldSchema(name="entity_count", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Community summaries with embeddings for hierarchical retrieval",
        enable_dynamic_field=True
    )
    
    collection_name = "communities"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️  Dropped existing collection: {collection_name}")
    
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection: {collection_name}")
    
    # Vector index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 64}  # Fewer clusters for smaller community dataset
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Created vector index for {collection_name}")
    
    # Scalar indexes
    collection.create_index(field_name="level")
    collection.create_index(field_name="importance_score")
    print(f"✅ Created scalar indexes for {collection_name}")
    
    return collection

def create_claims_collection():
    """Create collection for claim embeddings"""
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="confidence_score", dtype=DataType.DOUBLE),
        FieldSchema(name="source_refs", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="validated", dtype=DataType.BOOL),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description="Factual claims with embeddings for fact-based retrieval",
        enable_dynamic_field=True
    )
    
    collection_name = "claims"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️  Dropped existing collection: {collection_name}")
    
    collection = Collection(collection_name, schema)
    print(f"✅ Created collection: {collection_name}")
    
    # Vector index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"✅ Created vector index for {collection_name}")
    
    # Scalar indexes
    collection.create_index(field_name="confidence_score")
    collection.create_index(field_name="validated")
    print(f"✅ Created scalar indexes for {collection_name}")
    
    return collection

def load_collections():
    """Load all collections into memory for querying"""
    collection_names = ["text_chunks", "entities", "communities", "claims"]
    
    for name in collection_names:
        if utility.has_collection(name):
            collection = Collection(name)
            collection.load()
            print(f"✅ Loaded collection into memory: {name}")
        else:
            print(f"⚠️  Collection not found: {name}")

def main():
    """Main setup function"""
    print("🚀 Setting up Milvus collections for AI Agent Platform Hybrid RAG")
    
    if not connect_milvus():
        sys.exit(1)
    
    try:
        # Create all collections
        text_chunks = create_text_chunks_collection()
        entities = create_entities_collection()
        communities = create_communities_collection()
        claims = create_claims_collection()
        
        print("\n📊 Collection Summary:")
        collections = [text_chunks, entities, communities, claims]
        for collection in collections:
            print(f"  • {collection.name}: {collection.num_entities} entities")
        
        # Load collections into memory
        print("\n💾 Loading collections into memory...")
        load_collections()
        
        print("\n🎉 Milvus setup completed successfully!")
        print("\nCollection Details:")
        print("  • text_chunks: Document chunks with 1536-dim embeddings")
        print("  • entities: Named entities with semantic embeddings")  
        print("  • communities: Entity communities with hierarchical structure")
        print("  • claims: Factual claims with validation status")
        print("\nAll collections use COSINE similarity for semantic search")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()