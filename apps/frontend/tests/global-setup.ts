/**
 * Global setup for Playwright E2E tests
 * Handles authentication, database seeding, and test environment preparation
 */
import { chromium, FullConfig } from '@playwright/test';
import path from 'path';

async function globalSetup(config: FullConfig) {
  console.log('🔧 Starting global test setup...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for backend to be ready
    console.log('⏳ Waiting for backend to be ready...');
    await page.goto('http://localhost:8000/health', { waitUntil: 'networkidle' });
    
    // Authenticate as test user
    console.log('🔐 Authenticating test user...');
    await page.goto('http://localhost:3000/login');
    
    // Fill login form
    await page.fill('[data-testid="email"]', 'testexample.com');
    await page.fill('[data-testid="password"]', 'test_password');
    await page.click('[data-testid="login-button"]');
    
    // Wait for authentication
    await page.waitForURL('**/dashboard', { timeout: 10000 });
    
    // Save authentication state
    const authFile = path.join(__dirname, 'auth.json');
    await page.context().storageState({ path: authFile });
    console.log(`✅ Authentication state saved to: ${authFile}`);
    
    // Seed test data
    console.log('🌱 Seeding test data...');
    await seedTestData(page);
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await browser.close();
  }
  
  console.log('✅ Global test setup completed');
}

async function seedTestData(page: any) {
  // Create test project
  await page.goto('http://localhost:3000/projects');
  
  try {
    await page.click('[data-testid="create-project-button"]', { timeout: 5000 });
    await page.fill('[data-testid="project-title"]', 'E2E Test Project');
    await page.fill('[data-testid="project-description"]', 'Project created for end-to-end testing');
    await page.click('[data-testid="save-project-button"]');
    
    // Wait for project creation
    await page.waitForSelector('[data-testid="project-card"]', { timeout: 10000 });
    console.log('✅ Test project created');
  } catch (error) {
    console.log('⚠️ Test project may already exist or creation failed:', error.message);
  }
  
  // Create test PRD
  try {
    await page.click('[data-testid="create-prd-button"]', { timeout: 5000 });
    await page.fill('[data-testid="prd-title"]', 'E2E Test PRD');
    await page.fill('[data-testid="prd-description"]', 'PRD created for end-to-end testing');
    await page.click('[data-testid="generate-prd-button"]');
    
    // Wait for PRD generation
    await page.waitForSelector('[data-testid="prd-content"]', { timeout: 15000 });
    console.log('✅ Test PRD created');
  } catch (error) {
    console.log('⚠️ Test PRD creation failed or may already exist:', error.message);
  }
}

export default globalSetup;