# Multi-stage build for Nuxt.js 4 with TypeScript
FROM node:20-alpine AS base

# Set working directory
WORKDIR /app

# Install pnpm for better performance
RUN corepack enable && corepack prepare pnpm@latest --activate

# Install system dependencies
RUN apk add --no-cache \
    dumb-init \
    curl \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S nodejs \
    && adduser -S nuxt -u 1001

# Development stage
FROM base AS development

# Copy package files
COPY package*.json pnpm-lock.yaml* ./
COPY shared/package*.json ./shared/

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source code
COPY . .
COPY ../shared ./shared

# Change ownership
RUN chown -R nuxt:nodejs /app
USER nuxt

# Expose ports
EXPOSE 3000 24678

# Start development server
CMD ["pnpm", "dev", "--host", "0.0.0.0"]

# Build stage
FROM base AS builder

# Set build environment
ENV NODE_ENV=production
ENV NUXT_TELEMETRY_DISABLED=1

# Copy package files
COPY package*.json pnpm-lock.yaml* ./
COPY shared/package*.json ./shared/

# Install dependencies (including dev dependencies for build)
RUN pnpm install --frozen-lockfile

# Copy source code and shared types
COPY . .
COPY ../shared ./shared

# Build the application
RUN pnpm build

# Prune dev dependencies
RUN pnpm prune --prod --no-optional

# Production stage
FROM base AS production

# Set production environment
ENV NODE_ENV=production
ENV NUXT_TELEMETRY_DISABLED=1
ENV NITRO_PORT=3000
ENV NITRO_HOST=0.0.0.0

# Copy built application from builder stage
COPY --from=builder --chown=nuxt:nodejs /app/.output /app/.output
COPY --from=builder --chown=nuxt:nodejs /app/node_modules /app/node_modules
COPY --from=builder --chown=nuxt:nodejs /app/package.json /app/package.json

# Copy static assets
COPY --chown=nuxt:nodejs ./public /app/public

# Create necessary directories
RUN mkdir -p /app/logs && chown nuxt:nodejs /app/logs

# Switch to non-root user
USER nuxt

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Expose port
EXPOSE 3000

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the application
CMD ["node", ".output/server/index.mjs"]

# Testing stage
FROM builder AS test

# Install test dependencies if not already present
RUN pnpm install --frozen-lockfile

# Copy test configuration files
COPY vitest.config.ts playwright.config.ts ./
COPY tests/ ./tests/

# Run tests
CMD ["pnpm", "test"]