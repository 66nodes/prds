/**
 * Global teardown for Playwright E2E tests
 * Handles cleanup of test data and test environment reset
 */
import { chromium, FullConfig } from '@playwright/test';
import fs from 'fs/promises';
import path from 'path';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global test teardown...');
  
  try {
    // Clean up authentication files
    const authFile = path.join(__dirname, 'auth.json');
    try {
      await fs.unlink(authFile);
      console.log('✅ Authentication state file cleaned up');
    } catch (error) {
      console.log('ℹ️ No authentication file to clean up');
    }
    
    // Clean up test data
    await cleanupTestData();
    
    // Clean up test reports and artifacts
    await cleanupTestArtifacts();
    
  } catch (error) {
    console.error('❌ Global teardown encountered error:', error);
    // Don't fail the tests if cleanup fails
  }
  
  console.log('✅ Global test teardown completed');
}

async function cleanupTestData() {
  console.log('🧹 Cleaning up test data...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Attempt to clean up via API if available
    await page.goto('http://localhost:8000/test/cleanup', { 
      waitUntil: 'networkidle',
      timeout: 5000 
    });
    console.log('✅ Test data cleaned up via API');
  } catch (error) {
    console.log('ℹ️ API cleanup not available or failed, skipping test data cleanup');
  } finally {
    await browser.close();
  }
}

async function cleanupTestArtifacts() {
  console.log('🧹 Cleaning up test artifacts...');
  
  const artifactPaths = [
    path.join(__dirname, '..', 'test-results'),
    path.join(__dirname, '..', 'playwright-report'),
    path.join(__dirname, '..', 'reports')
  ];
  
  for (const artifactPath of artifactPaths) {
    try {
      const stats = await fs.stat(artifactPath);
      if (stats.isDirectory()) {
        // Keep directories but clean up old files if needed
        const files = await fs.readdir(artifactPath);
        console.log(`ℹ️ Found ${files.length} files in ${artifactPath}`);
      }
    } catch (error) {
      // Directory doesn't exist, which is fine
      console.log(`ℹ️ Artifact directory ${artifactPath} not found`);
    }
  }
}

export default globalTeardown;