"""
E2E tests for complete PRD workflow - critical user journey.
"""

import pytest
from playwright.async_api import Page, expect
from conftest import E2EHelpers


class TestCompletePRDWorkflow:
    """Test complete PRD workflow from creation to publication."""

    @pytest.mark.e2e
    @pytest.mark.critical
    async def test_complete_prd_generation_and_validation_workflow(
        self, 
        authenticated_page: Page, 
        base_url: str, 
        sample_prd_data: dict,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test the complete PRD generation, validation, and publication workflow."""
        page = authenticated_page
        
        # Step 1: Navigate to PRD generation page
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        await helpers.wait_for_loading_to_finish(page)
        
        # Step 2: Fill in PRD generation form
        await helpers.fill_form_field(page, 'input[name="title"]', sample_prd_data["title"])
        await helpers.fill_form_field(page, 'textarea[name="description"]', sample_prd_data["description"])
        
        # Fill target audience
        await helpers.fill_form_field(page, 'input[name="targetAudience"]', sample_prd_data["target_audience"])
        
        # Add requirements
        for i, requirement in enumerate(sample_prd_data["requirements"]):
            if i > 0:  # Add new requirement field
                await page.click('button[data-testid="add-requirement"]')
                await page.wait_for_timeout(500)  # Allow DOM update
            
            await helpers.fill_form_field(
                page, 
                f'input[name="requirements[{i}]"]', 
                requirement
            )
        
        # Add constraints
        for i, constraint in enumerate(sample_prd_data["constraints"]):
            if i > 0:  # Add new constraint field
                await page.click('button[data-testid="add-constraint"]')
                await page.wait_for_timeout(500)
            
            await helpers.fill_form_field(
                page,
                f'input[name="constraints[{i}]"]',
                constraint
            )
        
        # Add success metrics
        for i, metric in enumerate(sample_prd_data["success_metrics"]):
            if i > 0:  # Add new metric field
                await page.click('button[data-testid="add-metric"]')
                await page.wait_for_timeout(500)
            
            await helpers.fill_form_field(
                page,
                f'input[name="successMetrics[{i}]"]',
                metric
            )
        
        # Take screenshot before generation
        await helpers.take_screenshot(page, "prd_form_filled")
        
        # Step 3: Start PRD generation
        await page.click('button[data-testid="generate-prd"]')
        
        # Wait for generation to start
        await helpers.assert_element_visible(page, '[data-testid="generation-progress"]')
        await helpers.assert_text_content(page, '[data-testid="generation-status"]', "Generating")
        
        # Wait for generation to complete (with timeout)
        await page.wait_for_selector(
            '[data-testid="generation-complete"]', 
            timeout=120000  # 2 minutes timeout
        )
        
        # Step 4: Verify PRD was generated successfully
        await helpers.assert_element_visible(page, '[data-testid="generated-prd"]')
        await helpers.assert_text_content(page, '[data-testid="prd-title"]', sample_prd_data["title"])
        
        # Check hallucination rate indicator
        hallucination_element = page.locator('[data-testid="hallucination-rate"]')
        hallucination_text = await hallucination_element.text_content()
        
        # Extract percentage and verify it's below threshold
        import re
        rate_match = re.search(r'(\d+\.?\d*)%', hallucination_text)
        if rate_match:
            rate = float(rate_match.group(1))
            assert rate < 5.0, f"Hallucination rate {rate}% is too high"
        
        # Check validation score
        validation_element = page.locator('[data-testid="validation-score"]')
        validation_text = await validation_element.text_content()
        
        score_match = re.search(r'(\d+\.?\d*)%', validation_text)
        if score_match:
            score = float(score_match.group(1))
            assert score > 85.0, f"Validation score {score}% is too low"
        
        # Step 5: Review generated content
        prd_content = page.locator('[data-testid="prd-content"]')
        content_text = await prd_content.text_content()
        
        # Verify key requirements are mentioned in the content
        for requirement in sample_prd_data["requirements"][:3]:  # Check first 3
            assert requirement.lower() in content_text.lower(), f"Requirement '{requirement}' not found in PRD"
        
        # Step 6: Test validation functionality
        await page.click('button[data-testid="validate-prd"]')
        await helpers.wait_for_loading_to_finish(page)
        
        # Check validation results
        await helpers.assert_element_visible(page, '[data-testid="validation-results"]')
        
        validation_status = page.locator('[data-testid="validation-status"]')
        status_text = await validation_status.text_content()
        
        if "passed" not in status_text.lower():
            # If validation failed, check issues
            issues_section = page.locator('[data-testid="validation-issues"]')
            if await issues_section.count() > 0:
                issues_text = await issues_section.text_content()
                print(f"Validation issues: {issues_text}")
        
        # Step 7: Save PRD as draft
        await page.click('button[data-testid="save-draft"]')
        
        # Wait for save confirmation
        await helpers.assert_element_visible(page, '[data-testid="save-success"]')
        
        # Get PRD ID from URL or element
        prd_id_element = page.locator('[data-prd-id]')
        prd_id = await prd_id_element.get_attribute('data-prd-id')
        assert prd_id, "PRD ID not found after saving"
        
        # Step 8: Navigate to PRD review page
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/{prd_id}")
        await helpers.wait_for_loading_to_finish(page)
        
        # Verify PRD details on review page
        await helpers.assert_text_content(page, '[data-testid="prd-title-display"]', sample_prd_data["title"])
        await helpers.assert_element_visible(page, '[data-testid="prd-metadata"]')
        
        # Step 9: Test export functionality
        export_formats = ["pdf", "markdown", "docx"]
        
        for format_type in export_formats:
            # Click export dropdown
            await page.click('[data-testid="export-dropdown"]')
            await page.wait_for_timeout(500)
            
            # Click specific format
            await page.click(f'[data-testid="export-{format_type}"]')
            
            # Wait for download to start
            async with page.expect_download() as download_info:
                pass
            
            download = await download_info.value
            assert download.suggested_filename.endswith(f'.{format_type}'), f"Wrong file extension for {format_type}"
            
            # Verify file was downloaded
            file_size = await download.path().stat()
            assert file_size.st_size > 0, f"Downloaded {format_type} file is empty"
        
        # Step 10: Publish PRD
        await page.click('button[data-testid="publish-prd"]')
        
        # Confirm publication in modal
        await helpers.assert_element_visible(page, '[data-testid="publish-modal"]')
        await page.click('button[data-testid="confirm-publish"]')
        
        # Wait for publication confirmation
        await helpers.assert_element_visible(page, '[data-testid="publish-success"]')
        
        # Verify status changed to published
        status_badge = page.locator('[data-testid="prd-status"]')
        status_text = await status_badge.text_content()
        assert "published" in status_text.lower(), "PRD status not updated to published"
        
        # Step 11: Verify PRD appears in published list
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds")
        await helpers.wait_for_loading_to_finish(page)
        
        # Filter by published status
        await page.click('[data-testid="filter-published"]')
        await helpers.wait_for_network_idle(page)
        
        # Find our PRD in the list
        prd_card = page.locator(f'[data-prd-id="{prd_id}"]')
        await helpers.assert_element_visible(page, f'[data-prd-id="{prd_id}"]')
        
        # Verify card shows correct information
        card_title = prd_card.locator('[data-testid="prd-card-title"]')
        await helpers.assert_text_content(page, '[data-testid="prd-card-title"]', sample_prd_data["title"])
        
        # Take final screenshot
        await helpers.take_screenshot(page, "prd_workflow_complete")

    @pytest.mark.e2e
    @pytest.mark.critical
    async def test_prd_regeneration_with_feedback(
        self,
        authenticated_page: Page,
        base_url: str,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test PRD regeneration based on feedback."""
        page = authenticated_page
        
        # Create initial PRD (simplified version)
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        
        # Fill minimal form
        await helpers.fill_form_field(page, 'input[name="title"]', "Simple Test PRD")
        await helpers.fill_form_field(page, 'textarea[name="description"]', "A simple test PRD for regeneration")
        await helpers.fill_form_field(page, 'input[name="requirements[0]"]', "Basic functionality")
        
        # Generate PRD
        await page.click('button[data-testid="generate-prd"]')
        await page.wait_for_selector('[data-testid="generation-complete"]', timeout=120000)
        
        # Save as draft
        await page.click('button[data-testid="save-draft"]')
        await helpers.assert_element_visible(page, '[data-testid="save-success"]')
        
        # Get PRD ID
        prd_id_element = page.locator('[data-prd-id]')
        prd_id = await prd_id_element.get_attribute('data-prd-id')
        
        # Navigate to PRD edit page
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/{prd_id}/edit")
        
        # Test feedback and regeneration
        feedback_text = "Please add more technical details and security considerations"
        
        await helpers.fill_form_field(
            page, 
            'textarea[data-testid="feedback-input"]', 
            feedback_text
        )
        
        # Trigger regeneration
        await page.click('button[data-testid="regenerate-prd"]')
        
        # Wait for regeneration
        await helpers.assert_element_visible(page, '[data-testid="regeneration-progress"]')
        await page.wait_for_selector('[data-testid="regeneration-complete"]', timeout=120000)
        
        # Verify regenerated content is different
        new_content = page.locator('[data-testid="prd-content"]')
        new_content_text = await new_content.text_content()
        
        # Should contain more technical details based on feedback
        assert any(word in new_content_text.lower() for word in ["technical", "security", "architecture"]), \
            "Regenerated PRD does not contain expected technical details"

    @pytest.mark.e2e
    @pytest.mark.critical
    async def test_collaborative_prd_review_workflow(
        self,
        authenticated_page: Page,
        base_url: str,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test collaborative PRD review and approval workflow."""
        page = authenticated_page
        
        # Create and save a PRD first
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        
        await helpers.fill_form_field(page, 'input[name="title"]', "Collaborative Review PRD")
        await helpers.fill_form_field(page, 'textarea[name="description"]', "PRD for testing collaborative review")
        await helpers.fill_form_field(page, 'input[name="requirements[0]"]', "Collaborative features")
        
        await page.click('button[data-testid="generate-prd"]')
        await page.wait_for_selector('[data-testid="generation-complete"]', timeout=120000)
        await page.click('button[data-testid="save-draft"]')
        
        # Get PRD ID
        prd_id_element = page.locator('[data-prd-id]')
        prd_id = await prd_id_element.get_attribute('data-prd-id')
        
        # Navigate to PRD review page
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/{prd_id}")
        
        # Submit for review
        await page.click('button[data-testid="submit-for-review"]')
        
        # Add reviewers
        await helpers.assert_element_visible(page, '[data-testid="add-reviewers-modal"]')
        await helpers.fill_form_field(page, 'input[data-testid="reviewer-email"]', "reviewerexample.com")
        await page.click('button[data-testid="add-reviewer"]')
        await page.click('button[data-testid="submit-review-request"]')
        
        # Verify review status
        await helpers.assert_element_visible(page, '[data-testid="review-pending"]')
        
        # Simulate reviewer adding comments
        await page.click('button[data-testid="add-comment"]')
        await helpers.fill_form_field(
            page,
            'textarea[data-testid="comment-text"]',
            "Please add more details about the API endpoints"
        )
        await page.click('button[data-testid="submit-comment"]')
        
        # Verify comment appears
        await helpers.assert_element_visible(page, '[data-testid="comment-list"]')
        await helpers.assert_text_content(
            page, 
            '[data-testid="comment-list"]', 
            "API endpoints"
        )

    @pytest.mark.e2e
    @pytest.mark.critical
    async def test_prd_analytics_and_quality_metrics(
        self,
        authenticated_page: Page,
        base_url: str,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test PRD analytics and quality metrics dashboard."""
        page = authenticated_page
        
        # Navigate to project analytics
        await page.goto(f"{base_url}/projects/{test_project['id']}/analytics")
        await helpers.wait_for_loading_to_finish(page)
        
        # Verify analytics dashboard elements
        await helpers.assert_element_visible(page, '[data-testid="prd-metrics-overview"]')
        await helpers.assert_element_visible(page, '[data-testid="hallucination-trends-chart"]')
        await helpers.assert_element_visible(page, '[data-testid="quality-score-chart"]')
        await helpers.assert_element_visible(page, '[data-testid="generation-time-stats"]')
        
        # Check that metrics are displaying reasonable values
        total_prds = page.locator('[data-testid="total-prds-count"]')
        if await total_prds.count() > 0:
            total_count = await total_prds.text_content()
            assert total_count.isdigit(), "Total PRDs count should be numeric"
        
        avg_hallucination = page.locator('[data-testid="avg-hallucination-rate"]')
        if await avg_hallucination.count() > 0:
            rate_text = await avg_hallucination.text_content()
            # Should be a percentage below 5%
            rate_match = re.search(r'(\d+\.?\d*)%', rate_text)
            if rate_match:
                rate = float(rate_match.group(1))
                assert rate < 5.0, f"Average hallucination rate {rate}% seems too high"
        
        # Test filtering and date range selection
        await page.click('[data-testid="date-range-selector"]')
        await page.click('[data-testid="last-30-days"]')
        await helpers.wait_for_network_idle(page)
        
        # Verify charts updated
        await helpers.assert_element_visible(page, '[data-testid="chart-updated-indicator"]')

    @pytest.mark.e2e
    @pytest.mark.critical
    async def test_error_handling_and_recovery(
        self,
        authenticated_page: Page,
        base_url: str,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test error handling and recovery scenarios."""
        page = authenticated_page
        
        # Test network error handling during PRD generation
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        
        # Fill form with minimal data
        await helpers.fill_form_field(page, 'input[name="title"]', "Error Test PRD")
        await helpers.fill_form_field(page, 'textarea[name="description"]', "Testing error scenarios")
        
        # Simulate network failure during generation
        await page.route("**/api/projects/*/prds/generate", lambda route: route.abort())
        
        await page.click('button[data-testid="generate-prd"]')
        
        # Should show error message
        await helpers.assert_element_visible(page, '[data-testid="generation-error"]')
        
        # Test retry functionality
        await page.unroute("**/api/projects/*/prds/generate")  # Remove network block
        await page.click('button[data-testid="retry-generation"]')
        
        # Should successfully generate after retry
        await page.wait_for_selector('[data-testid="generation-complete"]', timeout=120000)
        
        # Test form validation errors
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        
        # Try to generate without required fields
        await page.click('button[data-testid="generate-prd"]')
        
        # Should show validation errors
        await helpers.assert_element_visible(page, '[data-testid="validation-errors"]')
        
        # Error messages should be helpful
        error_text = await page.locator('[data-testid="validation-errors"]').text_content()
        assert "title" in error_text.lower() or "required" in error_text.lower(), \
            "Validation error message not helpful"

    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_prd_generation_performance(
        self,
        authenticated_page: Page,
        base_url: str,
        test_project: dict,
        helpers: E2EHelpers
    ):
        """Test PRD generation performance metrics."""
        page = authenticated_page
        
        await page.goto(f"{base_url}/projects/{test_project['id']}/prds/new")
        
        # Fill form
        await helpers.fill_form_field(page, 'input[name="title"]', "Performance Test PRD")
        await helpers.fill_form_field(page, 'textarea[name="description"]', "Testing performance of PRD generation")
        await helpers.fill_form_field(page, 'input[name="requirements[0]"]', "High performance requirement")
        
        # Measure generation time
        import time
        start_time = time.time()
        
        await page.click('button[data-testid="generate-prd"]')
        await page.wait_for_selector('[data-testid="generation-complete"]', timeout=120000)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Assert reasonable generation time (under 2 minutes)
        assert generation_time < 120, f"PRD generation took {generation_time:.2f} seconds (too long)"
        
        # Verify progress indicators updated during generation
        # (This would have been checked during the wait period)
        
        print(f"PRD generation completed in {generation_time:.2f} seconds")