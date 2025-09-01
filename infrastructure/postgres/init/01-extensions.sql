-- AI Agent Platform - PostgreSQL Extensions and Basic Setup
-- Enable necessary extensions for the platform

-- Core PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Performance and monitoring
CREATE EXTENSION IF NOT EXISTS "pg_buffercache";

-- Create application database if it doesn't exist
SELECT 'CREATE DATABASE aiplatform' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'aiplatform');

-- Set up basic configuration
ALTER DATABASE aiplatform SET timezone TO 'UTC';

-- Performance tuning for AI workloads
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;