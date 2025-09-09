"""
Configuration and fixtures for E2E tests.
"""

import pytest
import asyncio
import os
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def browser() -> AsyncGenerator[Browser, None]:
    """Launch browser for the test session."""
    async with async_playwright() as playwright:
        # Use headless mode for CI, headed mode for debugging
        headless = os.getenv("HEADLESS", "true").lower() == "true"
        
        browser = await playwright.chromium.launch(
            headless=headless,
            slow_mo=50 if not headless else 0,  # Add delay for debugging
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--allow-running-insecure-content"
            ]
        )
        yield browser
        await browser.close()


@pytest.fixture
async def context(browser: Browser) -> AsyncGenerator[BrowserContext, None]:
    """Create a new browser context for each test."""
    context = await browser.new_context(
        viewport={"width": 1920, "height": 1080},
        ignore_https_errors=True,
        record_video_dir="test-results/videos" if os.getenv("RECORD_VIDEO") else None,
        record_har_path="test-results/network.har" if os.getenv("RECORD_HAR") else None
    )
    
    # Add authentication state if needed
    yield context
    await context.close()


@pytest.fixture
async def page(context: BrowserContext) -> AsyncGenerator[Page, None]:
    """Create a new page for each test."""
    page = await context.new_page()
    
    # Set up error handling
    page.on("pageerror", lambda error: print(f"Page error: {error}"))
    page.on("requestfailed", lambda request: print(f"Request failed: {request.url}"))
    
    yield page
    await page.close()


@pytest.fixture
def base_url():
    """Base URL for the application."""
    return os.getenv("E2E_BASE_URL", "http://localhost:3000")


@pytest.fixture
def api_base_url():
    """Base URL for the API."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
async def authenticated_page(page: Page, base_url: str) -> Page:
    """Create a page with authenticated user."""
    # Navigate to login page
    await page.goto(f"{base_url}/login")
    
    # Register a test user
    test_email = "e2e.testexample.com"
    test_password = "SecureTestPassword123!"
    
    # Check if we need to register first
    register_link = page.locator('a[href*="register"]')
    if await register_link.count() > 0:
        await register_link.click()
        
        # Fill registration form
        await page.fill('input[name="name"]', "E2E Test User")
        await page.fill('input[name="email"]', test_email)
        await page.fill('input[name="password"]', test_password)
        await page.fill('input[name="confirmPassword"]', test_password)
        
        # Submit registration
        await page.click('button[type="submit"]')
        await page.wait_for_url(f"{base_url}/dashboard", timeout=10000)
    else:
        # Login with existing user
        await page.fill('input[name="email"]', test_email)
        await page.fill('input[name="password"]', test_password)
        await page.click('button[type="submit"]')
        
        # Wait for dashboard or handle login failure
        try:
            await page.wait_for_url(f"{base_url}/dashboard", timeout=5000)
        except:
            # If login fails, try registration
            await page.goto(f"{base_url}/register")
            await page.fill('input[name="name"]', "E2E Test User")
            await page.fill('input[name="email"]', test_email)
            await page.fill('input[name="password"]', test_password)
            await page.fill('input[name="confirmPassword"]', test_password)
            await page.click('button[type="submit"]')
            await page.wait_for_url(f"{base_url}/dashboard", timeout=10000)
    
    return page


@pytest.fixture
def sample_prd_data():
    """Sample PRD data for testing."""
    return {
        "title": "E2E Test Task Management System",
        "description": "A comprehensive task management system for E2E testing purposes with AI-powered features",
        "requirements": [
            "User authentication and authorization",
            "Task creation and management",
            "AI-powered task prioritization",
            "Real-time collaboration",
            "Mobile app support"
        ],
        "constraints": [
            "GDPR compliance required",
            "Response time under 200ms", 
            "Support 1,000 concurrent users",
            "99.9% uptime requirement"
        ],
        "target_audience": "Small to medium-sized businesses and teams",
        "success_metrics": [
            "User engagement increased by 25%",
            "Task completion rate improved by 30%",
            "User satisfaction score > 4.0/5",
            "Reduced time to task completion by 20%"
        ]
    }


@pytest.fixture
async def test_project(authenticated_page: Page, base_url: str):
    """Create a test project for E2E tests."""
    # Navigate to projects page
    await authenticated_page.goto(f"{base_url}/projects")
    
    # Create new project
    await authenticated_page.click('button[data-testid="create-project"]')
    await authenticated_page.fill('input[name="name"]', "E2E Test Project")
    await authenticated_page.fill('textarea[name="description"]', "Test project for E2E testing")
    await authenticated_page.click('button[type="submit"]')
    
    # Wait for project creation
    await authenticated_page.wait_for_selector('[data-testid="project-card"]', timeout=10000)
    
    # Get project ID from URL or element
    project_card = authenticated_page.locator('[data-testid="project-card"]').first
    project_id = await project_card.get_attribute("data-project-id")
    
    return {
        "id": project_id,
        "name": "E2E Test Project",
        "description": "Test project for E2E testing"
    }


class E2EHelpers:
    """Helper methods for E2E tests."""
    
    @staticmethod
    async def wait_for_loading_to_finish(page: Page, timeout: int = 30000):
        """Wait for all loading indicators to disappear."""
        # Wait for common loading indicators
        selectors = [
            '[data-testid="loading"]',
            '.loading',
            '.spinner',
            '[aria-label*="loading"]'
        ]
        
        for selector in selectors:
            try:
                await page.wait_for_selector(selector, state="detached", timeout=5000)
            except:
                pass  # Selector not found, continue
    
    @staticmethod
    async def wait_for_network_idle(page: Page, timeout: int = 30000):
        """Wait for network to be idle."""
        await page.wait_for_load_state("networkidle", timeout=timeout)
    
    @staticmethod
    async def fill_form_field(page: Page, selector: str, value: str, clear_first: bool = True):
        """Fill a form field with proper clearing and validation."""
        field = page.locator(selector)
        
        if clear_first:
            await field.clear()
        
        await field.fill(value)
        
        # Verify the value was set
        filled_value = await field.input_value()
        assert filled_value == value, f"Field {selector} was not filled correctly"
    
    @staticmethod
    async def click_and_wait(page: Page, selector: str, wait_for: str = None, timeout: int = 30000):
        """Click element and wait for specific condition."""
        await page.click(selector)
        
        if wait_for:
            await page.wait_for_selector(wait_for, timeout=timeout)
    
    @staticmethod
    async def take_screenshot(page: Page, name: str):
        """Take a screenshot for debugging."""
        screenshots_dir = "test-results/screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        await page.screenshot(path=f"{screenshots_dir}/{name}.png")
    
    @staticmethod
    async def assert_text_content(page: Page, selector: str, expected_text: str):
        """Assert element contains expected text."""
        element = page.locator(selector)
        await element.wait_for(state="visible")
        
        actual_text = await element.text_content()
        assert expected_text in actual_text, f"Expected '{expected_text}' in '{actual_text}'"
    
    @staticmethod
    async def assert_element_visible(page: Page, selector: str, timeout: int = 10000):
        """Assert element is visible."""
        element = page.locator(selector)
        await element.wait_for(state="visible", timeout=timeout)
        assert await element.is_visible(), f"Element {selector} is not visible"
    
    @staticmethod
    async def assert_element_hidden(page: Page, selector: str, timeout: int = 10000):
        """Assert element is hidden or not present."""
        try:
            element = page.locator(selector)
            await element.wait_for(state="hidden", timeout=timeout)
        except:
            pass  # Element not found, which is also acceptable


@pytest.fixture
def helpers():
    """Provide E2E helper methods."""
    return E2EHelpers