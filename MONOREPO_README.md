# PRDS Monorepo

This is a monorepo for the PRDS (AI-Powered Strategic Planning Platform) project, managed with pnpm
workspaces.

## Project Structure

```
prds/
├── apps/
│   ├── frontend/          # Nuxt.js 4 frontend application
│   └── backend/           # FastAPI backend application
├── packages/
│   ├── eslint-config/     # Shared ESLint configuration
│   └── prettier-config/   # Shared Prettier configuration
├── pnpm-workspace.yaml    # pnpm workspace configuration
├── package.json           # Root package.json with workspace scripts
├── tsconfig.json          # Root TypeScript configuration
├── .eslintrc.js           # Root ESLint configuration
├── .prettierrc.js         # Root Prettier configuration
└── README.md              # Project documentation
```

## Prerequisites

- Node.js >= 18.0.0
- pnpm >= 8.0.0
- Python >= 3.11 (for backend)

## Getting Started

1. **Install dependencies:**

   ```bash
   pnpm install
   ```

2. **Development:**

   ```bash
   # Start frontend and backend in parallel
   pnpm dev

   # Start only frontend
   pnpm dev:frontend

   # Start only backend
   pnpm dev:backend
   ```

3. **Building:**

   ```bash
   # Build all projects
   pnpm build

   # Build specific project
   pnpm build:frontend
   pnpm build:backend
   ```

4. **Testing:**

   ```bash
   # Test all projects
   pnpm test

   # Test specific project
   pnpm test:frontend
   pnpm test:backend
   ```

5. **Linting:**

   ```bash
   # Lint all projects
   pnpm lint

   # Lint specific project
   pnpm lint:frontend
   pnpm lint:backend
   ```

6. **Formatting:**

   ```bash
   # Format all files
   pnpm format

   # Check formatting
   pnpm format:check
   ```

## Workspace Scripts

- `pnpm dev` - Start all projects in development mode
- `pnpm build` - Build all projects
- `pnpm test` - Run tests for all projects
- `pnpm lint` - Lint all projects
- `pnpm format` - Format all files with Prettier
- `pnpm install:all` - Install dependencies for all projects

## Code Quality Tools

- **ESLint**: Code linting with TypeScript and Vue.js support
- **Prettier**: Code formatting
- **TypeScript**: Strict type checking for frontend
- **Husky**: Pre-commit hooks (to be configured)

## Adding New Projects

1. Create a new directory in `apps/` or `packages/`
2. Add a `package.json` with a unique name
3. Update `pnpm-workspace.yaml` if needed
4. Add scripts to root `package.json`

## Contributing

1. Ensure all tests pass: `pnpm test`
2. Ensure code is properly formatted: `pnpm format`
3. Ensure code passes linting: `pnpm lint`
4. Commit your changes
