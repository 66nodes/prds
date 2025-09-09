import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test('should display login form', async ({ page }) => {
    await page.goto('/login');

    await expect(page.locator('h2')).toContainText('Sign in to your account');
    await expect(page.locator('input[type="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('button[type="submit"]')).toContainText(
      'Sign in'
    );
  });

  test('should show validation errors for empty form', async ({ page }) => {
    await page.goto('/login');

    await page.click('button[type="submit"]');

    // HTML5 validation should prevent form submission
    const emailInput = page.locator('input[type="email"]');
    await expect(emailInput).toHaveAttribute('required');
  });

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/login');

    await page.click('text=create a new account');
    await expect(page).toHaveURL('/register');
  });

  test('should redirect to dashboard when authenticated', async ({ page }) => {
    // Mock authenticated state
    await page.addInitScript(() => {
      localStorage.setItem(
        'auth',
        JSON.stringify({
          user: { id: '1', email: 'testexample.com', name: 'Test User' },
          isAuthenticated: true,
        })
      );
    });

    await page.goto('/login');
    await expect(page).toHaveURL('/dashboard');
  });
});

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authenticated state
    await page.addInitScript(() => {
      localStorage.setItem(
        'auth',
        JSON.stringify({
          user: { id: '1', email: 'testexample.com', name: 'Test User' },
          isAuthenticated: true,
        })
      );
    });
  });

  test('should display dashboard content', async ({ page }) => {
    await page.goto('/dashboard');

    await expect(page.locator('h1')).toContainText('Dashboard');
    await expect(page.locator('nav[aria-label="Tabs"]')).toBeVisible();
    await expect(page.locator('text=Overview')).toBeVisible();
  });

  test('should switch between tabs', async ({ page }) => {
    await page.goto('/dashboard');

    // Click Projects tab
    await page.click('text=Projects');
    await expect(page.locator('text=Projects')).toHaveClass(
      /border-indigo-500/
    );

    // Click PRDs tab
    await page.click('text=PRDs');
    await expect(page.locator('text=PRDs')).toHaveClass(/border-indigo-500/);
  });

  test('should show user menu', async ({ page }) => {
    await page.goto('/dashboard');

    // Click user avatar
    await page.click('button:has-text("TU")');
    await expect(page.locator('text=Your Profile')).toBeVisible();
    await expect(page.locator('text=Settings')).toBeVisible();
    await expect(page.locator('text=Sign out')).toBeVisible();
  });
});
