import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json', 'lcov', 'clover'],
      reportsDirectory: './coverage',
      exclude: [
        'coverage/**',
        'dist/**',
        'packages/*/test{,s}/**',
        '**/*.d.ts',
        'cypress/**',
        'test{,s}/**',
        'test{,-*}.{js,cjs,mjs,ts,tsx,jsx}',
        '**/*{.,-}test.{js,cjs,mjs,ts,tsx,jsx}',
        '**/*{.,-}spec.{js,cjs,mjs,ts,tsx,jsx}',
        '**/__tests__/**',
        '**/{karma,rollup,webpack,vite,vitest,jest,ava,babel,nyc,cypress,tsup,build}.config.*',
        '**/.{eslint,mocha,prettier}rc.{js,cjs,yml}',
        'nuxt.config.*',
        'app.vue',
        'error.vue',
        '**/*.config.*',
        'plugins/**',
        'middleware/**',
        'layouts/**',
        'server/**',
        'assets/**',
        'public/**'
      ],
      all: true,
      clean: true,
      statements: 90,
      branches: 85,
      functions: 90,
      lines: 90,
      skipFull: false,
      watermarks: {
        statements: [80, 90],
        functions: [80, 90],
        branches: [75, 85],
        lines: [80, 90]
      }
    },
    outputFile: {
      junit: './reports/junit.xml',
      html: './reports/html/index.html'
    },
    silent: false,
    reporter: ['default', 'junit'],
    logHeapUsage: true,
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: false
      }
    }
  },
  resolve: {
    alias: {
      '~': resolve(__dirname, '.'),
      '@': resolve(__dirname, '.'),
      '#app': resolve(__dirname, 'node_modules/nuxt/dist/app'),
      '#imports': resolve(__dirname, '.nuxt/imports'),
    },
  },
});
