/**
 * End-to-End tests for Human Validation Workflow
 */

import { test, expect } from '@playwright/test';

test.describe('Human Validation Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authentication
    await page.route('**/auth/me', async route => {
      await route.fulfill({
        json: {
          id: 'test-user-123',
          email: 'testexample.com',
          name: 'Test User'
        }
      });
    });

    // Mock WebSocket connection
    await page.addInitScript(() => {
      (window as any).WebSocket = class MockWebSocket {
        constructor(url: string) {
          this.url = url;
          this.readyState = 1; // OPEN
          setTimeout(() => {
            this.onopen?.({});
          }, 100);
        }
        
        send(data: string) {
          // Echo back validation requests for testing
          const parsed = JSON.parse(data);
          if (parsed.type === 'human_validation_request') {
            setTimeout(() => {
              this.onmessage?.({
                data: JSON.stringify({
                  type: 'human_validation_request',
                  payload: {
                    id: 'test-validation-123',
                    conversation_id: 'test-conv-123',
                    prompt: {
                      id: 'test-validation-123',
                      type: 'approval',
                      question: 'Do you approve this approach?',
                      context: 'Testing human validation workflow',
                      required: true,
                      timeout: 30000
                    }
                  }
                })
              });
            }, 500);
          }
        }
        
        close() {}
        
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent) => void) | null = null;
        onclose: ((event: CloseEvent) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;
      };
    });

    // Navigate to conversation page
    await page.goto('/conversation');
  });

  test('should display validation prompt when human input is required', async ({ page }) => {
    // Mock API responses
    await page.route('**/human-validation/request', async route => {
      await route.fulfill({
        json: {
          validation_id: 'test-validation-123',
          status: 'requested',
          message: 'Validation request created successfully'
        }
      });
    });

    // Type a message that triggers validation
    await page.fill('[placeholder*="Describe your project"]', 'I need a complex AI system with human validation');
    await page.click('[icon="i-heroicons-paper-airplane"]');

    // Wait for validation prompt to appear
    await expect(page.locator('.validation-prompt')).toBeVisible({ timeout: 10000 });
    
    // Verify validation prompt content
    await expect(page.locator('.validation-prompt')).toContainText('Human Input Required');
    await expect(page.locator('.validation-prompt')).toContainText('Do you approve this approach?');
    await expect(page.locator('.validation-prompt')).toContainText('Testing human validation workflow');
  });

  test('should handle approval validation correctly', async ({ page }) => {
    // Mock validation response API
    await page.route('**/human-validation/respond', async route => {
      const request = await route.request();
      const body = await request.postDataJSON();
      
      expect(body.approved).toBe(true);
      expect(body.validation_id).toBe('test-validation-123');
      
      await route.fulfill({
        json: {
          validation_id: 'test-validation-123',
          status: 'completed',
          message: 'Validation response submitted successfully'
        }
      });
    });

    // Trigger validation prompt (simplified)
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-validation-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-validation-123',
            type: 'approval',
            question: 'Do you approve this approach?',
            context: 'Testing human validation workflow',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    // Wait for validation prompt
    await expect(page.locator('.validation-prompt')).toBeVisible();

    // Add feedback
    await page.fill('textarea[placeholder*="Optional feedback"]', 'This approach looks good to me');

    // Click approve button
    await page.click('button:has-text("Approve")');

    // Verify validation prompt disappears
    await expect(page.locator('.validation-prompt')).not.toBeVisible();

    // Verify success feedback
    await expect(page.locator('text=✅ Approved')).toBeVisible();
  });

  test('should handle rejection validation correctly', async ({ page }) => {
    // Mock validation response API
    await page.route('**/human-validation/respond', async route => {
      const request = await route.request();
      const body = await request.postDataJSON();
      
      expect(body.approved).toBe(false);
      expect(body.feedback).toContain('needs revision');
      
      await route.fulfill({
        json: {
          validation_id: 'test-validation-123',
          status: 'completed',
          message: 'Validation response submitted successfully'
        }
      });
    });

    // Trigger validation prompt
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-validation-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-validation-123',
            type: 'approval',
            question: 'Do you approve this approach?',
            context: 'Testing human validation workflow',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();

    // Add rejection feedback
    await page.fill('textarea[placeholder*="Optional feedback"]', 'This approach needs revision');

    // Click reject button
    await page.click('button:has-text("Reject")');

    // Verify validation prompt disappears
    await expect(page.locator('.validation-prompt')).not.toBeVisible();

    // Verify rejection feedback
    await expect(page.locator('text=❌ Needs revision')).toBeVisible();
  });

  test('should handle choice validation correctly', async ({ page }) => {
    // Mock choice validation response
    await page.route('**/human-validation/respond', async route => {
      const request = await route.request();
      const body = await request.postDataJSON();
      
      expect(body.approved).toBe(true);
      expect(body.response.choice).toBe('option_b');
      
      await route.fulfill({
        json: {
          validation_id: 'test-choice-123',
          status: 'completed',
          message: 'Choice validation submitted successfully'
        }
      });
    });

    // Trigger choice validation prompt
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-choice-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-choice-123',
            type: 'choice',
            question: 'Which implementation approach should we use?',
            context: 'Please select the best approach for the project',
            required: true,
            options: [
              { label: 'Option A: REST API', value: 'option_a', description: 'Traditional REST API approach' },
              { label: 'Option B: GraphQL', value: 'option_b', description: 'Modern GraphQL approach' }
            ]
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();
    await expect(page.locator('text=Which implementation approach should we use?')).toBeVisible();

    // Select option B
    await page.click('input[value="option_b"]');

    // Submit choice
    await page.click('button:has-text("Submit Choice")');

    // Verify validation completed
    await expect(page.locator('.validation-prompt')).not.toBeVisible();
    await expect(page.locator('text=✅ Approved')).toBeVisible();
  });

  test('should handle validation timeout correctly', async ({ page }) => {
    // Trigger validation with short timeout
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-timeout-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-timeout-123',
            type: 'approval',
            question: 'Do you approve this approach?',
            context: 'Testing timeout validation',
            required: true,
            timeout: 2000 // 2 seconds for testing
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();

    // Verify timeout indicator is visible
    await expect(page.locator('text=Time remaining:')).toBeVisible();

    // Wait for timeout (mock will auto-reject)
    await page.waitForTimeout(3000);

    // Verify timeout handling
    await expect(page.locator('.validation-prompt')).not.toBeVisible();
  });

  test('should handle input validation correctly', async ({ page }) => {
    // Mock input validation response
    await page.route('**/human-validation/respond', async route => {
      const request = await route.request();
      const body = await request.postDataJSON();
      
      expect(body.approved).toBe(true);
      expect(body.response.input).toContain('additional requirements');
      
      await route.fulfill({
        json: {
          validation_id: 'test-input-123',
          status: 'completed',
          message: 'Input validation submitted successfully'
        }
      });
    });

    // Trigger input validation prompt
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-input-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-input-123',
            type: 'input',
            question: 'Please provide additional requirements',
            context: 'We need more details about the user interface requirements',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();
    await expect(page.locator('text=Please provide additional requirements')).toBeVisible();

    // Fill in input
    const inputArea = page.locator('textarea[placeholder*="Please provide your input"]');
    await inputArea.fill('Here are the additional requirements: responsive design, accessibility compliance, and mobile-first approach');

    // Submit input
    await page.click('button:has-text("Submit")');

    // Verify validation completed
    await expect(page.locator('.validation-prompt')).not.toBeVisible();
    await expect(page.locator('text=✅ Approved')).toBeVisible();
  });

  test('should display validation history', async ({ page }) => {
    // Mock validation history API
    await page.route('**/human-validation/history', async route => {
      await route.fulfill({
        json: [
          {
            id: 'validation-1',
            type: 'approval',
            conversation_id: 'conv-1',
            user_id: 'test-user-123',
            request_data: { question: 'First validation question' },
            response_data: { approved: true, feedback: 'Looks good' },
            status: 'completed',
            created_at: '2024-01-01T10:00:00Z',
            updated_at: '2024-01-01T10:01:00Z',
            expires_at: null
          },
          {
            id: 'validation-2',
            type: 'choice',
            conversation_id: 'conv-1',
            user_id: 'test-user-123',
            request_data: { question: 'Second validation question' },
            response_data: { approved: true, choice: 'option_a' },
            status: 'completed',
            created_at: '2024-01-01T11:00:00Z',
            updated_at: '2024-01-01T11:01:00Z',
            expires_at: null
          }
        ]
      });
    });

    // Navigate to validation history (if there's a dedicated page/section)
    // Or trigger display of validation points
    await page.click('[icon="i-heroicons-check-circle"]');

    // Verify validation history is displayed
    await expect(page.locator('text=Validations (2)')).toBeVisible();
  });

  test('should prevent message input during validation', async ({ page }) => {
    // Trigger validation prompt
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-block-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-block-123',
            type: 'approval',
            question: 'Do you approve this approach?',
            context: 'Testing input blocking during validation',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();

    // Verify message input is disabled
    const messageInput = page.locator('[placeholder*="Describe your project"]');
    await expect(messageInput).toBeDisabled();

    // Verify send button is disabled
    const sendButton = page.locator('[icon="i-heroicons-paper-airplane"]');
    await expect(sendButton).toBeDisabled();
  });

  test('should handle validation cancellation', async ({ page }) => {
    // Mock cancellation API
    await page.route('**/human-validation/test-cancel-123', async route => {
      if (route.request().method() === 'DELETE') {
        await route.fulfill({
          json: {
            validation_id: 'test-cancel-123',
            status: 'cancelled',
            message: 'Validation request cancelled successfully'
          }
        });
      }
    });

    // Trigger validation prompt with cancel option
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-cancel-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-cancel-123',
            type: 'confirmation',
            question: 'I understand the implications of this action',
            context: 'Please confirm before proceeding',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();

    // Click cancel button
    await page.click('button:has-text("Cancel")');

    // Verify validation was cancelled
    await expect(page.locator('.validation-prompt')).not.toBeVisible();
  });
});

test.describe('Validation API Integration', () => {
  test('should handle API errors gracefully', async ({ page }) => {
    // Mock API error
    await page.route('**/human-validation/respond', async route => {
      await route.fulfill({
        status: 500,
        json: {
          detail: 'Internal server error'
        }
      });
    });

    // Trigger validation and attempt submission
    await page.evaluate(() => {
      const event = new CustomEvent('human_validation_request', {
        detail: {
          id: 'test-error-123',
          conversation_id: 'test-conv-123',
          prompt: {
            id: 'test-error-123',
            type: 'approval',
            question: 'Test error handling?',
            context: 'Testing API error handling',
            required: true
          }
        }
      });
      document.dispatchEvent(event);
    });

    await expect(page.locator('.validation-prompt')).toBeVisible();
    await page.click('button:has-text("Approve")');

    // Verify error handling
    await expect(page.locator('text=Failed to submit validation')).toBeVisible();
    
    // Validation prompt should still be visible for retry
    await expect(page.locator('.validation-prompt')).toBeVisible();
  });
});