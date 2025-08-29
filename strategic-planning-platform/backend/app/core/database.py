from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
from typing import AsyncGenerator, Optional
import asyncio

from app.core.config import settings


class Base(DeclarativeBase):
    pass


# PostgreSQL connection
engine = create_async_engine(
    settings.async_postgres_url,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Neo4j connection
class Neo4jConnection:
    def __init__(self):
        self.driver: Optional[AsyncGraphDatabase.driver] = None
    
    async def init(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        await self.verify_connectivity()
        await self.create_constraints()
    
    async def verify_connectivity(self):
        if self.driver:
            try:
                await self.driver.verify_connectivity()
                print("Neo4j connection verified")
            except Exception as e:
                print(f"Neo4j connection failed: {e}")
                raise
    
    async def create_constraints(self):
        if not self.driver:
            return
            
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PRD) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Objective) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE",
        ]
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (r:Requirement) ON (r.embedding)",
            "CREATE FULLTEXT INDEX requirement_search IF NOT EXISTS FOR (r:Requirement) ON EACH [r.description, r.acceptance_criteria]",
            "CREATE INDEX IF NOT EXISTS FOR (p:PRD) ON (p.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Task) ON (t.status)",
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.created_at)",
        ]
        
        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    print(f"Constraint creation warning: {e}")
            
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    print(f"Index creation warning: {e}")
    
    async def close(self):
        if self.driver:
            await self.driver.close()
    
    async def get_session(self):
        if self.driver:
            return self.driver.session()
        raise RuntimeError("Neo4j driver not initialized")


neo4j_db = Neo4jConnection()


async def get_neo4j():
    return neo4j_db


# Redis connections
redis_cache: Optional[Redis] = None
redis_session: Optional[Redis] = None


async def init_redis():
    global redis_cache, redis_session
    
    redis_cache = Redis.from_url(
        f"{settings.REDIS_URL.split('/')[0]}//{settings.REDIS_URL.split('//')[1].split('/')[0]}/{settings.REDIS_CACHE_DB}",
        decode_responses=True,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={},
        health_check_interval=30,
    )
    
    redis_session = Redis.from_url(
        f"{settings.REDIS_URL.split('/')[0]}//{settings.REDIS_URL.split('//')[1].split('/')[0]}/{settings.REDIS_SESSION_DB}",
        decode_responses=True,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={},
        health_check_interval=30,
    )
    
    # Test connections
    await redis_cache.ping()
    await redis_session.ping()
    print("Redis connections established")


async def get_redis_cache() -> Redis:
    if redis_cache is None:
        raise RuntimeError("Redis cache not initialized")
    return redis_cache


async def get_redis_session() -> Redis:
    if redis_session is None:
        raise RuntimeError("Redis session store not initialized")
    return redis_session


# Database initialization
async def init_postgresql():
    """Initialize PostgreSQL database and create tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("PostgreSQL initialized")


async def init_neo4j():
    """Initialize Neo4j connection and constraints"""
    await neo4j_db.init()
    print("Neo4j initialized")


async def init_databases():
    """Initialize all database connections"""
    await asyncio.gather(
        init_postgresql(),
        init_neo4j(),
        init_redis()
    )