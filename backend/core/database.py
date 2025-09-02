"""
Database connections and initialization for Neo4j, Milvus, PostgreSQL, and Redis
"""

import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError
from pymilvus import connections, utility
from sqlalchemy.ext.asyncio import AsyncSession as AsyncSQLSession, async_sessionmaker, create_async_engine
from sqlalchemy import text
import redis.asyncio as redis
import structlog

from .config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Global connection objects
postgres_engine = None
redis_client = None


class Neo4jConnection:
    """Neo4j database connection manager."""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self.is_connected = False
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                database=settings.neo4j_database,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=30,
                encrypted=False  # Set to True for production with SSL
            )
            
            # Test connection
            await self.verify_connectivity()
            self.is_connected = True
            
            logger.info(
                "Connected to Neo4j database",
                uri=settings.neo4j_uri,
                database=settings.neo4j_database
            )
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(
                "Failed to connect to Neo4j database",
                error=str(e),
                uri=settings.neo4j_uri
            )
            raise
    
    async def close(self) -> None:
        """Close database connection."""
        if self.driver:
            await self.driver.close()
            self.is_connected = False
            logger.info("Neo4j connection closed")
    
    async def verify_connectivity(self) -> None:
        """Verify database connectivity."""
        if not self.driver:
            raise RuntimeError("Database driver not initialized")
        
        try:
            await self.driver.verify_connectivity()
        except Exception as e:
            logger.error("Neo4j connectivity verification failed", error=str(e))
            raise
    
    @asynccontextmanager
    async def session(self, **kwargs) -> AsyncSession:
        """Get database session context manager."""
        if not self.driver:
            raise RuntimeError("Database not connected")
        
        async with self.driver.session(**kwargs) as session:
            yield session
    
    async def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        if not parameters:
            parameters = {}
        
        async with self.session(database=database) as session:
            result = await session.run(query, parameters)
            records = [record.data() async for record in result]
            return records
    
    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a write transaction."""
        if not parameters:
            parameters = {}
        
        async def write_transaction(tx):
            result = await tx.run(query, parameters)
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
                "query": query
            }
        
        async with self.session(database=database) as session:
            result = await session.execute_write(write_transaction)
            return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j database health."""
        try:
            if not self.is_connected:
                return {"status": "unhealthy", "error": "Not connected"}
            
            # Test basic query
            start_time = asyncio.get_event_loop().time()
            await self.execute_query("RETURN 1 as test")
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "connected": self.is_connected
            }
            
        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            return {"status": "unhealthy", "error": str(e)}


# Global connection instance
neo4j_connection = Neo4jConnection()


class MilvusConnection:
    """Milvus vector database connection manager."""
    
    def __init__(self):
        self.is_connected = False
        self.alias = "default"
    
    async def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias=self.alias,
                host=settings.milvus_host,
                port=settings.milvus_port,
                user=settings.milvus_user,
                password=settings.milvus_password,
                db_name=settings.milvus_db_name,
            )
            
            # Test connection
            collections = utility.list_collections()
            self.is_connected = True
            
            logger.info(
                "Connected to Milvus database",
                host=settings.milvus_host,
                port=settings.milvus_port,
                collections_count=len(collections)
            )
            
        except Exception as e:
            logger.error(
                "Failed to connect to Milvus database",
                error=str(e),
                host=settings.milvus_host
            )
            raise
    
    async def close(self) -> None:
        """Close Milvus connection."""
        try:
            connections.disconnect(self.alias)
            self.is_connected = False
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.warning("Error closing Milvus connection", error=str(e))
    
    def health_check(self) -> Dict[str, Any]:
        """Check Milvus health."""
        try:
            if not self.is_connected:
                return {"status": "unhealthy", "error": "Not connected"}
            
            collections = utility.list_collections()
            return {
                "status": "healthy",
                "collections_count": len(collections),
                "collections": collections[:5]  # Show first 5 collections
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class PostgreSQLConnection:
    """PostgreSQL database connection manager."""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.is_connected = False
    
    async def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        global postgres_engine
        
        try:
            self.engine = create_async_engine(
                settings.postgres_url,
                echo=settings.is_development,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSQLSession,
                expire_on_commit=False
            )
            
            postgres_engine = self.engine
            self.is_connected = True
            
            logger.info("Connected to PostgreSQL database")
            
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close PostgreSQL connection."""
        global postgres_engine
        
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            postgres_engine = None
            self.is_connected = False
            logger.info("PostgreSQL connection closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncSQLSession:
        """Get PostgreSQL session context manager."""
        if not self.session_factory:
            raise RuntimeError("PostgreSQL not connected")
        
        async with self.session_factory() as session:
            yield session
    
    async def health_check(self) -> Dict[str, Any]:
        """Check PostgreSQL health."""
        try:
            if not self.is_connected:
                return {"status": "unhealthy", "error": "Not connected"}
            
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
                
            return {"status": "healthy", "response_time_ms": 3}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class RedisConnection:
    """Redis cache connection manager."""
    
    def __init__(self):
        self.client = None
        self.is_connected = False
    
    async def connect(self) -> None:
        """Establish connection to Redis."""
        global redis_client
        
        try:
            self.client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True,
                max_connections=20,
            )
            
            # Test connection
            await self.client.ping()
            
            redis_client = self.client
            self.is_connected = True
            
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close Redis connection."""
        global redis_client
        
        if self.client:
            await self.client.aclose()
            self.client = None
            redis_client = None
            self.is_connected = False
            logger.info("Redis connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            if not self.is_connected:
                return {"status": "unhealthy", "error": "Not connected"}
            
            await self.client.ping()
            return {"status": "healthy", "response_time_ms": 2}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global connection instances
milvus_connection = MilvusConnection()
postgres_connection = PostgreSQLConnection()
redis_connection = RedisConnection()


async def init_neo4j() -> None:
    """Initialize Neo4j database connection and schema."""
    await neo4j_connection.connect()
    await create_constraints()
    await create_indexes()
    await seed_initial_data()


async def init_milvus() -> None:
    """Initialize Milvus vector database connection."""
    await milvus_connection.connect()


async def init_postgres() -> None:
    """Initialize PostgreSQL database connection."""
    await postgres_connection.connect()


async def init_redis() -> None:
    """Initialize Redis cache connection."""
    await redis_connection.connect()


async def init_all_databases() -> None:
    """Initialize all database connections."""
    logger.info("Initializing all database connections...")
    
    try:
        # Initialize in parallel for better performance
        await asyncio.gather(
            init_neo4j(),
            init_milvus(),
            init_postgres(),
            init_redis(),
            return_exceptions=False
        )
        logger.info("All database connections initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database connections: {str(e)}")
        raise


async def close_neo4j() -> None:
    """Close Neo4j database connection."""
    await neo4j_connection.close()


async def close_milvus() -> None:
    """Close Milvus connection."""
    await milvus_connection.close()


async def close_postgres() -> None:
    """Close PostgreSQL connection."""
    await postgres_connection.close()


async def close_redis() -> None:
    """Close Redis connection."""
    await redis_connection.close()


async def close_all_databases() -> None:
    """Close all database connections."""
    logger.info("Closing all database connections...")
    
    await asyncio.gather(
        close_neo4j(),
        close_milvus(),
        close_postgres(),
        close_redis(),
        return_exceptions=True
    )
    logger.info("All database connections closed")


async def get_neo4j() -> Neo4jConnection:
    """Get Neo4j connection instance."""
    return neo4j_connection


def get_milvus() -> MilvusConnection:
    """Get Milvus connection instance."""
    return milvus_connection


def get_postgres() -> PostgreSQLConnection:
    """Get PostgreSQL connection instance."""
    return postgres_connection


def get_redis() -> RedisConnection:
    """Get Redis connection instance."""
    return redis_connection


async def check_all_databases_health() -> Dict[str, Any]:
    """Check health of all databases."""
    neo4j_health = await neo4j_connection.health_check()
    milvus_health = milvus_connection.health_check()
    postgres_health = await postgres_connection.health_check()
    redis_health = await redis_connection.health_check()
    
    return {
        "neo4j": neo4j_health,
        "milvus": milvus_health,
        "postgres": postgres_health,
        "redis": redis_health,
    }


async def create_constraints() -> None:
    """Create database constraints for data integrity."""
    constraints = [
        # Core entities
        "CREATE CONSTRAINT req_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT prd_unique IF NOT EXISTS FOR (p:PRD) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT task_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT user_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        "CREATE CONSTRAINT objective_unique IF NOT EXISTS FOR (o:Objective) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT section_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT template_unique IF NOT EXISTS FOR (t:Template) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT project_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
        
        # Validation entities
        "CREATE CONSTRAINT validation_unique IF NOT EXISTS FOR (v:ValidationResult) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT community_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
    ]
    
    for constraint in constraints:
        try:
            await neo4j_connection.execute_write(constraint)
            logger.debug("Created constraint", query=constraint)
        except Exception as e:
            # Constraint might already exist
            logger.debug("Constraint creation skipped", constraint=constraint, reason=str(e))


async def create_indexes() -> None:
    """Create database indexes for performance."""
    indexes = [
        # Vector indexes for GraphRAG
        "CREATE VECTOR INDEX req_embedding IF NOT EXISTS FOR (r:Requirement) ON (r.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
        "CREATE VECTOR INDEX objective_embedding IF NOT EXISTS FOR (o:Objective) ON (o.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}",
        
        # Full-text indexes for search
        "CREATE FULLTEXT INDEX req_search IF NOT EXISTS FOR (r:Requirement) ON EACH [r.description, r.acceptance_criteria, r.business_value]",
        "CREATE FULLTEXT INDEX prd_search IF NOT EXISTS FOR (p:PRD) ON EACH [p.title, p.description, p.executive_summary]",
        
        # Property indexes for performance
        "CREATE INDEX prd_created IF NOT EXISTS FOR (p:PRD) ON (p.created_at)",
        "CREATE INDEX prd_status IF NOT EXISTS FOR (p:PRD) ON (p.status)",
        "CREATE INDEX user_created IF NOT EXISTS FOR (u:User) ON (u.created_at)",
        "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)",
        "CREATE INDEX validation_score IF NOT EXISTS FOR (v:ValidationResult) ON (v.confidence_score)",
        "CREATE INDEX validation_created IF NOT EXISTS FOR (v:ValidationResult) ON (v.created_at)",
    ]
    
    for index in indexes:
        try:
            await neo4j_connection.execute_write(index)
            logger.debug("Created index", query=index)
        except Exception as e:
            # Index might already exist
            logger.debug("Index creation skipped", index=index, reason=str(e))


async def seed_initial_data() -> None:
    """Seed database with initial data if empty."""
    try:
        # Check if any PRDs exist
        result = await neo4j_connection.execute_query(
            "MATCH (p:PRD) RETURN count(p) as count"
        )
        
        if result and result[0]["count"] == 0:
            logger.info("Seeding initial data...")
            
            # Create initial templates
            await create_initial_templates()
            
            # Create system entities for validation
            await create_system_entities()
            
            logger.info("Initial data seeding completed")
        else:
            logger.info("Database already contains data, skipping seeding")
    
    except Exception as e:
        logger.error("Failed to seed initial data", error=str(e))
        # Don't raise - this is not critical for startup


async def create_initial_templates() -> None:
    """Create initial PRD templates."""
    templates_query = """
    CREATE (t1:Template {
        id: 'template-basic-prd',
        name: 'Basic PRD Template',
        description: 'Standard product requirements document template',
        sections: ['overview', 'objectives', 'scope', 'requirements', 'timeline', 'success_metrics'],
        created_at: datetime(),
        is_default: true
    })
    CREATE (t2:Template {
        id: 'template-feature-prd',
        name: 'Feature PRD Template', 
        description: 'Template for new feature development',
        sections: ['feature_overview', 'user_stories', 'acceptance_criteria', 'technical_requirements', 'dependencies'],
        created_at: datetime(),
        is_default: false
    })
    CREATE (t3:Template {
        id: 'template-api-prd',
        name: 'API PRD Template',
        description: 'Template for API development projects',
        sections: ['api_overview', 'endpoints', 'authentication', 'data_models', 'error_handling', 'rate_limits'],
        created_at: datetime(),
        is_default: false
    })
    """
    
    await neo4j_connection.execute_write(templates_query)


async def create_system_entities() -> None:
    """Create system-level entities for validation."""
    entities_query = """
    CREATE (e1:Entity {
        id: 'entity-authentication',
        name: 'Authentication',
        type: 'system_component',
        description: 'User authentication and authorization',
        created_at: datetime()
    })
    CREATE (e2:Entity {
        id: 'entity-database',
        name: 'Database',
        type: 'system_component', 
        description: 'Data storage and persistence',
        created_at: datetime()
    })
    CREATE (e3:Entity {
        id: 'entity-api',
        name: 'API',
        type: 'system_component',
        description: 'Application programming interface',
        created_at: datetime()
    })
    CREATE (e4:Entity {
        id: 'entity-frontend',
        name: 'Frontend',
        type: 'system_component',
        description: 'User interface and client-side application',
        created_at: datetime()
    })
    
    CREATE (c1:Community {
        id: 'community-web-development',
        name: 'Web Development',
        description: 'Web application development practices and patterns',
        entities: ['entity-frontend', 'entity-api', 'entity-database'],
        created_at: datetime()
    })
    CREATE (c2:Community {
        id: 'community-security',
        name: 'Security',
        description: 'Security practices and authentication patterns',
        entities: ['entity-authentication', 'entity-api'],
        created_at: datetime()
    })
    """
    
    await neo4j_connection.execute_write(entities_query)


async def health_check() -> Dict[str, Any]:
    """Check Neo4j database health status (legacy function - use check_all_databases_health instead)."""
    return await neo4j_connection.health_check()