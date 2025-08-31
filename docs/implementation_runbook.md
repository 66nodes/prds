# Implementation Runbook: AI-Powered Strategic Planning Platform

## Project Structure and Setup Instructions

```markdown
# PROJECT: AI-Powered Strategic Planning Platform
# TECH STACK: Nuxt.js 4 + FastAPI + Neo4j + GraphRAG
# PURPOSE: Build hallucination-free PRD generation system

## Directory Structure to Create:
strategic-planning-platform/
├── frontend/                 # Nuxt.js 4 application
│   ├── components/
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── prd/
│   │   └── shared/
│   ├── pages/
│   ├── composables/
│   ├── stores/
│   ├── assets/
│   └── server/
├── backend/                  # Python FastAPI services
│   ├── api/
│   │   ├── endpoints/
│   │   ├── middleware/
│   │   └── dependencies/
│   ├── services/
│   │   ├── graphrag/
│   │   ├── llm/
│   │   └── document/
│   ├── models/
│   ├── schemas/
│   └── core/
├── infrastructure/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
└── tests/
```

## Phase 1: Environment Setup (Day 1-2)

### Step 1.1: Initialize Frontend (Nuxt.js 4)

```bash
# Create Nuxt.js 4 project
npx nuxi@latest init frontend
cd frontend

# Install required dependencies
npm install @nuxt/ui @nuxtjs/tailwindcss @pinia/nuxt
npm install -D @types/node typescript vue-tsc
npm install @vueuse/nuxt @nuxtjs/google-fonts

# Create nuxt.config.ts
```

```typescript
// frontend/nuxt.config.ts
export default defineNuxtConfig({
  modules: [
    '@nuxt/ui',
    '@nuxtjs/tailwindcss',
    '@pinia/nuxt',
    '@vueuse/nuxt',
    '@nuxtjs/google-fonts'
  ],
  devtools: { enabled: true },
  typescript: {
    strict: true,
    typeCheck: true
  },
  ui: {
    global: true,
    icons: ['heroicons', 'lucide']
  },
  css: ['~/assets/css/main.css', '~/assets/css/theme.css']
})
```

### Step 1.2: Configure Design System

```css
/* frontend/assets/css/theme.css */
@layer base {
  :root {
    /* Custom black scale */
    --color-black-50: #f7f7f7;
    --color-black-100: #e3e3e3;
    --color-black-200: #c8c8c8;
    --color-black-300: #a4a4a4;
    --color-black-400: #818181;
    --color-black-500: #666666;
    --color-black-600: #515151;
    --color-black-700: #434343;
    --color-black-800: #383838;
    --color-black-900: #313131;
    --color-black-950: #1a1a1a;
    
    /* Semantic colors */
    --color-primary: var(--color-black-700);
    --color-secondary: #6366f1;
    --color-success: #10b981;
    --color-warning: #f59e0b;
    --color-error: #f97316;
    --color-info: #0ea5e9;
  }
  
  .dark {
    --color-primary: var(--color-black-600);
    --ui-bg: #0b0b0b;
    --ui-foreground: #e5e7eb;
  }
}
```

```typescript
// frontend/tailwind.config.ts
import type { Config } from 'tailwindcss'

export default <Partial<Config>>{
  content: [],
  theme: {
    extend: {
      colors: {
        black: {
          50: 'var(--color-black-50)',
          100: 'var(--color-black-100)',
          200: 'var(--color-black-200)',
          300: 'var(--color-black-300)',
          400: 'var(--color-black-400)',
          500: 'var(--color-black-500)',
          600: 'var(--color-black-600)',
          700: 'var(--color-black-700)',
          800: 'var(--color-black-800)',
          900: 'var(--color-black-900)',
          950: 'var(--color-black-950)'
        }
      }
    }
  }
}
```

### Step 1.3: Initialize Backend (FastAPI)

```bash
# Create Python backend
cd ../
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt
```

```python
# backend/requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
supabase==2.3.0  # Supabase Python client
neo4j==5.16.0
redis==5.0.1
celery==5.3.4
openai==1.10.0
llama-index==0.9.40
llama-index-graph-stores-neo4j==0.1.2
chromadb==0.4.22
python-dotenv==1.0.0
pytest==7.4.4
pytest-asyncio==0.23.3
httpx==0.26.0
```

```bash
pip install -r backend/requirements.txt
```

### Step 1.4: Setup Supabase and Neo4j Databases

```bash
# Create infrastructure directory and Docker compose file
mkdir -p infrastructure/docker
cd infrastructure/docker
```

```yaml
# infrastructure/docker/docker-compose.yml
version: '3.8'

services:
  # Supabase Services (Self-hosted stack)
  supabase-db:
    image: supabase/postgres:15.1.0.117
    container_name: supabase-db
    ports:
      - "5432:5432"
    environment:
      POSTGRES_HOST: /var/run/postgresql
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your-super-secret-password
      JWT_SECRET: your-super-secret-jwt-token
      JWT_EXP: 3600
    volumes:
      - supabase_db_data:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U postgres -h localhost
      interval: 5s
      timeout: 5s
      retries: 10

  supabase-auth:
    image: supabase/gotrue:v2.132.3
    container_name: supabase-auth
    depends_on:
      - supabase-db
    ports:
      - "9999:9999"
    environment:
      GOTRUE_API_HOST: 0.0.0.0
      GOTRUE_API_PORT: 9999
      API_EXTERNAL_URL: http://localhost:9999
      GOTRUE_DB_DRIVER: postgres
      GOTRUE_DB_DATABASE_URL: postgres://postgres:your-super-secret-password@supabase-db:5432/postgres?search_path=auth
      GOTRUE_JWT_SECRET: your-super-secret-jwt-token
      GOTRUE_JWT_EXP: 3600
      GOTRUE_SITE_URL: http://localhost:3000
      GOTRUE_MAILER_AUTOCONFIRM: true
      GOTRUE_EXTERNAL_EMAIL_ENABLED: false

  supabase-realtime:
    image: supabase/realtime:v2.27.5
    container_name: supabase-realtime
    depends_on:
      - supabase-db
    ports:
      - "4000:4000"
    environment:
      DB_HOST: supabase-db
      DB_PORT: 5432
      DB_NAME: postgres
      DB_USER: postgres
      DB_PASSWORD: your-super-secret-password
      DB_SSL: "false"
      PORT: 4000
      JWT_SECRET: your-super-secret-jwt-token
      REPLICATION_MODE: RLS
      REPLICATION_POLL_INTERVAL: 100
      SECURE_CHANNELS: "true"
      SLOT_NAME: supabase_realtime_rls
      TEMPORARY_SLOT: "true"

  supabase-storage:
    image: supabase/storage-api:v0.43.11
    container_name: supabase-storage
    depends_on:
      - supabase-db
    ports:
      - "5000:5000"
    environment:
      ANON_KEY: your-anon-key
      SERVICE_KEY: your-service-key
      PROJECT_REF: local
      POSTGREST_URL: http://supabase-postgrest:3000
      PGRST_JWT_SECRET: your-super-secret-jwt-token
      DATABASE_URL: postgres://postgres:your-super-secret-password@supabase-db:5432/postgres
      FILE_SIZE_LIMIT: 52428800
      STORAGE_BACKEND: file
      FILE_STORAGE_BACKEND_PATH: /var/lib/storage
    volumes:
      - supabase_storage_data:/var/lib/storage

  supabase-postgrest:
    image: postgrest/postgrest:v11.2.2
    container_name: supabase-postgrest
    depends_on:
      - supabase-db
    ports:
      - "3000:3000"
    environment:
      PGRST_DB_URI: postgres://postgres:your-super-secret-password@supabase-db:5432/postgres
      PGRST_DB_SCHEMAS: public,storage,auth
      PGRST_DB_ANON_ROLE: anon
      PGRST_JWT_SECRET: your-super-secret-jwt-token

  # Neo4j for GraphRAG
  neo4j:
    image: neo4j:5.15-enterprise
    container_name: strategic-planning-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your-password-here
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
      - NEO4J_dbms_memory_heap_initial__size=2G
      - NEO4J_dbms_memory_heap_max__size=4G
      - NEO4J_dbms_memory_pagecache_size=2G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  # Redis for caching and queues
  redis:
    image: redis:7-alpine
    container_name: strategic-planning-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  supabase_db_data:
  supabase_storage_data:
  neo4j_data:
  neo4j_logs:
  redis_data:
```

```bash
cd infrastructure/docker
docker-compose up -d
```

## Phase 2: Core Backend Implementation (Day 3-7)

### Step 2.1: FastAPI Application Structure

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.endpoints import auth, prd, dashboard
from core.config import settings
from core.database import init_databases

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_databases()
    yield
    # Shutdown
    # Clean up resources

app = FastAPI(
    title="Strategic Planning Platform API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(prd.router, prefix="/api/prd", tags=["prd"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Step 2.2: Supabase and Neo4j Database Connections

```python
# backend/core/database.py
from neo4j import AsyncGraphDatabase
from supabase import create_client, Client
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        self.driver: Optional[AsyncGraphDatabase.driver] = None
    
    async def init(self):
        self.driver = AsyncGraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))
        )
        await self.create_constraints()
    
    async def create_constraints(self):
        async with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PRD) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE"
            ]
            for constraint in constraints:
                await session.run(constraint)
    
    async def close(self):
        if self.driver:
            await self.driver.close()

class SupabaseConnection:
    def __init__(self):
        self.client: Optional[Client] = None
    
    async def init(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL", "http://localhost:3000")
        supabase_key = os.getenv("SUPABASE_ANON_KEY", "your-anon-key")
        self.client = create_client(supabase_url, supabase_key)
    
    def get_client(self) -> Client:
        return self.client

# Database connections
neo4j_db = Neo4jConnection()
supabase_db = SupabaseConnection()

async def init_databases():
    """Initialize both Neo4j and Supabase connections"""
    await neo4j_db.init()
    await supabase_db.init()

# Legacy alias for Neo4j
db = neo4j_db

async def init_neo4j():
    await neo4j_db.init()
```

### Step 2.3: GraphRAG Service Implementation

```python
# backend/services/graphrag/graphrag_service.py
from typing import List, Dict, Any
import numpy as np
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.graph_stores import Neo4jGraphStore
import openai
import os

class GraphRAGService:
    def __init__(self):
        self.graph_store = Neo4jGraphStore(
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    async def validate_against_graph(self, content: str, context: Dict) -> Dict:
        """
        Multi-level validation against knowledge graph
        Returns confidence score and corrections
        """
        # Entity-level validation
        entity_validation = await self._validate_entities(content)
        
        # Community-level validation
        community_validation = await self._validate_communities(content)
        
        # Global validation
        global_validation = await self._validate_global(content)
        
        # Calculate weighted confidence
        confidence = (
            entity_validation['score'] * 0.5 +
            community_validation['score'] * 0.3 +
            global_validation['score'] * 0.2
        )
        
        return {
            'confidence': confidence,
            'entity_validation': entity_validation,
            'community_validation': community_validation,
            'global_validation': global_validation,
            'requires_correction': confidence < 0.8
        }
    
    async def _validate_entities(self, content: str) -> Dict:
        # Implementation for entity validation
        query = """
        MATCH (r:Requirement)
        WHERE r.description CONTAINS $content_snippet
        RETURN r.id, r.description, r.confidence_score
        LIMIT 10
        """
        # Execute and process
        return {'score': 0.9, 'matches': []}
    
    async def retrieve_context(self, query: str, project_id: str) -> List[Dict]:
        """
        Hybrid retrieval combining vector search and graph traversal
        """
        # Vector similarity search
        vector_results = await self._vector_search(query)
        
        # Graph traversal for relationships
        graph_results = await self._graph_traversal(query, project_id)
        
        # Merge and rank results
        return self._merge_results(vector_results, graph_results)
```

### Step 2.4: PRD Workflow Endpoints

```python
# backend/api/endpoints/prd.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from pydantic import BaseModel
from services.graphrag.graphrag_service import GraphRAGService
from services.llm.llm_service import LLMService
from core.database import supabase_db

router = APIRouter()

class PRDPhase0Input(BaseModel):
    initial_description: str
    user_id: str

class PRDPhase1Input(BaseModel):
    prd_id: str
    answers: Dict[str, str]

@router.post("/phase0/initiate")
async def initiate_prd(input: PRDPhase0Input):
    """Phase 0: Process initial project description"""
    graphrag = GraphRAGService()
    llm = LLMService()
    
    # Extract concepts
    concepts = await llm.extract_concepts(input.initial_description)
    
    # Find similar projects
    similar_projects = await graphrag.find_similar_projects(concepts)
    
    # Generate clarifying questions
    questions = await llm.generate_clarifying_questions(
        input.initial_description,
        similar_projects
    )
    
    # Create PRD session in Supabase
    prd_id = await create_prd_session(input.user_id, input.initial_description)
    
    # Store in Supabase for user session management
    supabase_client = supabase_db.get_client()
    await supabase_client.table('prd_sessions').insert({
        'id': prd_id,
        'user_id': input.user_id,
        'initial_description': input.initial_description,
        'status': 'phase0_complete'
    }).execute()
    
    return {
        "prd_id": prd_id,
        "questions": questions,
        "similar_projects": similar_projects
    }

@router.post("/phase1/clarify")
async def clarify_objectives(input: PRDPhase1Input):
    """Phase 1: Process clarification answers"""
    graphrag = GraphRAGService()
    
    # Validate answers against graph
    validations = []
    for question_id, answer in input.answers.items():
        validation = await graphrag.validate_against_graph(answer, {})
        validations.append({
            "question_id": question_id,
            "validation": validation,
            "confidence": validation['confidence']
        })
    
    # Store validated answers in Supabase
    await store_phase1_answers(input.prd_id, input.answers, validations)
    
    # Update PRD session status in Supabase
    supabase_client = supabase_db.get_client()
    await supabase_client.table('prd_sessions').update({
        'phase1_answers': input.answers,
        'validations': validations,
        'status': 'phase1_complete'
    }).eq('id', input.prd_id).execute()
    
    return {
        "prd_id": input.prd_id,
        "validations": validations,
        "ready_for_phase2": all(v['confidence'] > 0.7 for v in validations)
    }
```

## Phase 3: Frontend Implementation (Day 8-12)

### Step 3.1: Supabase Authentication Integration

First, configure Nuxt.js to use Supabase authentication:

```bash
# Install Supabase client for Nuxt.js
npm install @supabase/supabase-js @nuxtjs/supabase
```

```typescript
// frontend/nuxt.config.ts - Add Supabase module
export default defineNuxtConfig({
  modules: [
    '@nuxt/ui',
    '@nuxtjs/tailwindcss',
    '@pinia/nuxt',
    '@vueuse/nuxt',
    '@nuxtjs/google-fonts',
    '@nuxtjs/supabase'  // Add Supabase module
  ],
  supabase: {
    url: process.env.SUPABASE_URL,
    key: process.env.SUPABASE_ANON_KEY,
    redirectOptions: {
      login: '/auth/login',
      callback: '/auth/callback',
      exclude: ['/auth/*']
    }
  },
  // ... rest of config
})
```

```typescript
// frontend/composables/useSupabaseAuth.ts
export const useSupabaseAuth = () => {
  const supabase = useSupabaseClient()
  const user = useSupabaseUser()

  const signIn = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { data, error }
  }

  const signUp = async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    return { data, error }
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    return { error }
  }

  return {
    user,
    signIn,
    signUp,
    signOut,
  }
}
```

### Step 3.2: Authentication Components

```vue
<!-- frontend/components/auth/LoginForm.vue -->
<template>
  <UCard class="w-full max-w-md">
    <template #header>
      <h2 class="text-2xl font-bold text-black-900 dark:text-white">
        Sign In
      </h2>
    </template>
    
    <UForm :schema="schema" :state="state" @submit="onSubmit">
      <UFormGroup label="Email" name="email">
        <UInput 
          v-model="state.email" 
          type="email"
          placeholder="you@example.com"
        />
      </UFormGroup>
      
      <UFormGroup label="Password" name="password">
        <UInput 
          v-model="state.password" 
          type="password"
          placeholder="••••••••"
        />
      </UFormGroup>
      
      <UButton 
        type="submit" 
        block 
        class="mt-4"
        :loading="loading"
      >
        Sign In
      </UButton>
    </UForm>
  </UCard>
</template>

<script setup lang="ts">
import { z } from 'zod'
import type { FormSubmitEvent } from '#ui/types'

const schema = z.object({
  email: z.string().email('Invalid email'),
  password: z.string().min(8, 'Must be at least 8 characters')
})

type Schema = z.output<typeof schema>

const state = reactive({
  email: '',
  password: ''
})

const loading = ref(false)

const { signIn } = useSupabaseAuth()

async function onSubmit(event: FormSubmitEvent<Schema>) {
  loading.value = true
  try {
    const { data, error } = await signIn(event.data.email, event.data.password)
    
    if (error) {
      throw error
    }
    
    // Redirect to dashboard on successful login
    await navigateTo('/dashboard')
  } catch (error) {
    console.error('Login failed:', error)
  } finally {
    loading.value = false
  }
}
</script>
```

### Step 3.3: PRD Creation Workflow Components

```vue
<!-- frontend/components/prd/Phase0.vue -->
<template>
  <div class="max-w-4xl mx-auto p-6">
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-black-900 dark:text-white mb-2">
        Welcome to Strategic Planning
      </h1>
      <p class="text-black-600 dark:text-gray-300">
        Describe your project idea in a sentence or paragraph
      </p>
    </div>
    
    <UCard>
      <UTextarea
        v-model="projectDescription"
        :rows="6"
        placeholder="e.g., Build a new mobile app for our loyalty program..."
        class="w-full"
      />
      
      <div class="mt-6 flex justify-end">
        <UButton
          size="lg"
          @click="handleSubmit"
          :disabled="!projectDescription.trim()"
          :loading="loading"
        >
          Continue
          <UIcon name="i-heroicons-arrow-right" class="ml-2" />
        </UButton>
      </div>
    </UCard>
    
    <!-- Similar Projects Preview -->
    <div v-if="similarProjects.length > 0" class="mt-8">
      <h3 class="text-lg font-semibold mb-4">Similar Projects</h3>
      <div class="grid gap-4">
        <UCard 
          v-for="project in similarProjects" 
          :key="project.id"
          class="hover:shadow-lg transition-shadow"
        >
          <h4 class="font-medium">{{ project.title }}</h4>
          <p class="text-sm text-gray-600 mt-1">
            {{ project.description }}
          </p>
        </UCard>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const projectDescription = ref('')
const loading = ref(false)
const similarProjects = ref([])

async function handleSubmit() {
  loading.value = true
  try {
    const { data } = await $fetch('/api/prd/phase0/initiate', {
      method: 'POST',
      body: {
        initial_description: projectDescription.value,
        user_id: useAuthStore().user.id
      }
    })
    
    // Store PRD ID and navigate to Phase 1
    usePrdStore().setPrdId(data.prd_id)
    usePrdStore().setQuestions(data.questions)
    
    await navigateTo('/prd/phase1')
  } catch (error) {
    console.error('Failed to initiate PRD:', error)
  } finally {
    loading.value = false
  }
}
</script>
```

### Step 3.4: Dashboard Implementation

```vue
<!-- frontend/pages/dashboard.vue -->
<template>
  <div class="container mx-auto p-6">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold">Strategic Planning Dashboard</h1>
      <p class="text-gray-600 mt-2">
        Manage your PRDs and track progress
      </p>
    </div>
    
    <!-- Quick Actions -->
    <div class="mb-8">
      <UButton 
        size="lg" 
        @click="navigateTo('/prd/create')"
        icon="i-heroicons-plus"
      >
        Create New PRD
      </UButton>
    </div>
    
    <!-- Metrics Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <UCard>
        <div class="text-2xl font-bold">{{ metrics.active }}</div>
        <div class="text-gray-600">Active PRDs</div>
      </UCard>
      
      <UCard>
        <div class="text-2xl font-bold">{{ metrics.completed }}</div>
        <div class="text-gray-600">Completed</div>
      </UCard>
      
      <UCard>
        <div class="text-2xl font-bold">{{ metrics.quality }}%</div>
        <div class="text-gray-600">Avg Quality Score</div>
      </UCard>
      
      <UCard>
        <div class="text-2xl font-bold">{{ metrics.time }}h</div>
        <div class="text-gray-600">Avg Time Saved</div>
      </UCard>
    </div>
    
    <!-- PRD List -->
    <UCard>
      <template #header>
        <h2 class="text-xl font-semibold">Recent PRDs</h2>
      </template>
      
      <UTable 
        :rows="prds" 
        :columns="columns"
        :loading="loading"
      >
        <template #actions-data="{ row }">
          <UDropdown :items="getActions(row)">
            <UButton 
              color="gray" 
              variant="ghost" 
              icon="i-heroicons-ellipsis-horizontal-20-solid" 
            />
          </UDropdown>
        </template>
      </UTable>
    </UCard>
  </div>
</template>

<script setup lang="ts">
// Use Supabase for dashboard data with authentication
const user = useSupabaseUser()
const supabase = useSupabaseClient()

const { data: metrics } = await useLazyAsyncData('dashboard-metrics', async () => {
  if (!user.value) return null
  const { data } = await supabase.from('dashboard_metrics').select('*').eq('user_id', user.value.id).single()
  return data
})

const { data: prds, pending: loading } = await useLazyAsyncData('user-prds', async () => {
  if (!user.value) return []
  const { data } = await supabase.from('prd_sessions').select('*').eq('user_id', user.value.id).order('created_at', { ascending: false })
  return data
})

const columns = [
  { key: 'title', label: 'Title' },
  { key: 'status', label: 'Status' },
  { key: 'created_at', label: 'Created' },
  { key: 'quality_score', label: 'Quality' },
  { key: 'actions', label: '' }
]

function getActions(row: any) {
  return [
    [{
      label: 'View',
      icon: 'i-heroicons-eye',
      click: () => navigateTo(`/prd/${row.id}`)
    }],
    [{
      label: 'Edit',
      icon: 'i-heroicons-pencil',
      click: () => navigateTo(`/prd/${row.id}/edit`)
    }],
    [{
      label: 'Export',
      icon: 'i-heroicons-arrow-down-tray',
      click: () => exportPrd(row.id)
    }]
  ]
}
</script>
```

## Phase 4: GraphRAG Integration (Day 13-16)

### Step 4.1: Initialize Graph Schema

```bash
# Run this script to initialize Neo4j schema
```

```python
# backend/scripts/init_graph.py
import asyncio
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

async def init_graph_schema():
    driver = AsyncGraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    
    async with driver.session() as session:
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT req_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT prd_unique IF NOT EXISTS FOR (p:PRD) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT task_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT user_unique IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
            "CREATE CONSTRAINT objective_unique IF NOT EXISTS FOR (o:Objective) REQUIRE o.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            await session.run(constraint)
            print(f"Created: {constraint}")
        
        # Create indexes
        indexes = [
            "CREATE INDEX req_embedding IF NOT EXISTS FOR (r:Requirement) ON (r.embedding)",
            "CREATE FULLTEXT INDEX req_search IF NOT EXISTS FOR (r:Requirement) ON EACH [r.description, r.acceptance_criteria]",
            "CREATE INDEX prd_created IF NOT EXISTS FOR (p:PRD) ON (p.created_at)",
            "CREATE INDEX task_status IF NOT EXISTS FOR (t:Task) ON (t.status)"
        ]
        
        for index in indexes:
            await session.run(index)
            print(f"Created: {index}")
    
    await driver.close()
    print("Graph schema initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_graph_schema())
```

### Step 4.2: Implement Hallucination Prevention

```python
# backend/services/graphrag/hallucination_prevention.py
from typing import Dict, List, Any
import numpy as np

class HallucinationPrevention:
    def __init__(self, neo4j_driver, confidence_threshold=0.8):
        self.driver = neo4j_driver
        self.confidence_threshold = confidence_threshold
        
    async def validate_content(self, content: str, context: Dict) -> Dict:
        """
        Three-tier validation system
        """
        # Level 1: Entity validation (50% weight)
        entity_result = await self._entity_validation(content, context)
        
        # Level 2: Community validation (30% weight)
        community_result = await self._community_validation(content, context)
        
        # Level 3: Global validation (20% weight)
        global_result = await self._global_validation(content, context)
        
        # Calculate weighted confidence
        confidence = (
            entity_result['confidence'] * 0.5 +
            community_result['confidence'] * 0.3 +
            global_result['confidence'] * 0.2
        )
        
        # Determine if correction needed
        needs_correction = confidence < self.confidence_threshold
        
        corrections = []
        if needs_correction:
            corrections = await self._generate_corrections(
                content,
                entity_result,
                community_result,
                global_result
            )
        
        return {
            'confidence': confidence,
            'valid': not needs_correction,
            'entity_validation': entity_result,
            'community_validation': community_result,
            'global_validation': global_result,
            'corrections': corrections
        }
    
    async def _entity_validation(self, content: str, context: Dict) -> Dict:
        query = """
        MATCH (r:Requirement)
        WHERE r.project_id = $project_id
        WITH r, 
             apoc.text.similarity(r.description, $content) as similarity
        WHERE similarity > 0.7
        RETURN r.id, r.description, similarity
        ORDER BY similarity DESC
        LIMIT 10
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, {
                'project_id': context.get('project_id'),
                'content': content
            })
            
            matches = [record async for record in result]
            
            if matches:
                avg_similarity = np.mean([m['similarity'] for m in matches])
                return {
                    'confidence': avg_similarity,
                    'matches': matches,
                    'status': 'validated'
                }
            
            return {
                'confidence': 0.0,
                'matches': [],
                'status': 'no_matches'
            }
```

## Phase 5: Testing and Validation (Day 17-19)

### Step 5.1: Backend Tests

```python
# backend/tests/test_graphrag.py
import pytest
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_phase0_initiation():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/prd/phase0/initiate",
            json={
                "initial_description": "Build a mobile app for loyalty program",
                "user_id": "test-user-123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "prd_id" in data
        assert "questions" in data
        assert len(data["questions"]) >= 3

@pytest.mark.asyncio
async def test_hallucination_prevention():
    from services.graphrag.hallucination_prevention import HallucinationPrevention
    
    hp = HallucinationPrevention(driver, confidence_threshold=0.8)
    
    # Test with known valid content
    result = await hp.validate_content(
        "Implement user authentication with OAuth 2.0",
        {"project_id": "test-project"}
    )
    
    assert result['confidence'] > 0.7
    assert 'corrections' in result
```

### Step 5.2: Frontend Tests

```typescript
// frontend/tests/components/Phase0.test.ts
import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import Phase0 from '~/components/prd/Phase0.vue'

describe('Phase0 Component', () => {
  it('renders welcome message', () => {
    const wrapper = mount(Phase0)
    expect(wrapper.text()).toContain('Welcome to Strategic Planning')
  })
  
  it('enables continue button when text is entered', async () => {
    const wrapper = mount(Phase0)
    const textarea = wrapper.find('textarea')
    const button = wrapper.find('button')
    
    expect(button.attributes('disabled')).toBe('true')
    
    await textarea.setValue('Test project description')
    expect(button.attributes('disabled')).toBeUndefined()
  })
})
```

## Phase 6: Deployment (Day 20)

### Step 6.1: Docker Configuration

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/.output ./
EXPOSE 3000
CMD ["node", "server/index.mjs"]
```

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6.2: Kubernetes Deployment

```yaml
# infrastructure/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategic-planning-frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: strategic-planning/frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NITRO_PORT
          value: "3000"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strategic-planning-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: strategic-planning/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: neo4j-uri
```

## Validation Checklist

```markdown
## Pre-Deployment Validation

### Backend Checks
- [ ] All API endpoints return correct status codes
- [ ] GraphRAG validation achieves <2% hallucination rate
- [ ] Response times <200ms for simple queries
- [ ] Neo4j indexes created and optimized
- [ ] Authentication/authorization working
- [ ] Rate limiting configured
- [ ] Error handling comprehensive

### Frontend Checks
- [ ] All phases of PRD workflow functional
- [ ] Dark mode working correctly
- [ ] Responsive design verified
- [ ] Form validation working
- [ ] Loading states implemented
- [ ] Error messages user-friendly

### Integration Checks
- [ ] Frontend successfully calls backend APIs
- [ ] WebSocket connections stable
- [ ] File uploads working
- [ ] Export functionality operational
- [ ] Real-time updates functioning

### Performance Checks
- [ ] Page load time <2 seconds
- [ ] API response time <200ms p95
- [ ] Concurrent user testing (100+)
- [ ] Memory usage stable
- [ ] No memory leaks detected

### Security Checks
- [ ] JWT tokens properly validated
- [ ] SQL injection prevention verified
- [ ] XSS protection in place
- [ ] CORS properly configured
- [ ] Secrets management secure

## Post-Deployment Monitoring

### Metrics to Track
- Hallucination rate (target: <2%)
- API response times
- User session duration
- PRD completion rate
- Error rates by endpoint
- Database query performance
```

This runbook provides a complete, step-by-step implementation guide that can be followed sequentially to build the AI-Powered Strategic Planning Platform. Each section includes specific commands, code snippets, and validation steps to ensure successful implementation.
