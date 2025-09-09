import { test, expect } from '@playwright/test';

// Test data
const TEST_PROJECT = {
  id: 'test-project-123',
  name: 'E2E Test Project',
  description: 'Project for end-to-end testing',
  status: 'planning'
};

const TEST_USER = {
  email: 'testexample.com',
  password: 'testpassword123'
};

test.describe('Conversational Interface', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authentication and project data
    await page.route('**/api/auth/me', async route => {
      await route.fulfill({
        json: {
          id: 'user-123',
          email: TEST_USER.email,
          name: 'Test User'
        }
      });
    });

    await page.route('**/api/projects', async route => {
      await route.fulfill({
        json: {
          data: [TEST_PROJECT]
        }
      });
    });

    // Navigate to the conversation page
    await page.goto('/conversation');
  });

  test('should display project selection initially', async ({ page }) => {
    // Should show project selection card
    await expect(page.getByText('Select a Project')).toBeVisible();
    
    // Should display test project
    await expect(page.getByText(TEST_PROJECT.name)).toBeVisible();
    await expect(page.getByText(TEST_PROJECT.description)).toBeVisible();
  });

  test('should start conversation when project is selected', async ({ page }) => {
    // Click on the test project
    await page.getByText(TEST_PROJECT.name).click();

    // Should show conversation interface
    await expect(page.getByText('Planning Session')).toBeVisible();
    
    // Should show welcome message
    await expect(page.getByText('Welcome! I\'ll help you create')).toBeVisible();
    
    // Should show message input
    await expect(page.getByPlaceholder('Describe your project requirements')).toBeVisible();
  });

  test('should send and display user messages', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Mock WebSocket connection
    await page.evaluate(() => {
      // Mock WebSocket for testing
      (window as any).mockWebSocket = {
        send: (data: string) => {
          const message = JSON.parse(data);
          // Simulate AI response after user message
          if (message.type === 'conversation_message') {
            setTimeout(() => {
              window.dispatchEvent(new CustomEvent('websocket-message', {
                detail: {
                  type: 'conversation_message',
                  payload: {
                    content: 'Thank you for sharing that! Can you tell me more about your target audience?',
                    metadata: { type: 'followup_question' }
                  }
                }
              }));
            }, 500);
          }
        }
      };
    });

    const messageInput = page.getByPlaceholder('Describe your project requirements');
    const testMessage = 'I want to build a mobile app for tracking fitness goals';

    // Type and send message
    await messageInput.fill(testMessage);
    await messageInput.press('Enter');

    // Should display user message
    await expect(page.getByText(testMessage)).toBeVisible();
    await expect(page.getByText('You')).toBeVisible();

    // Should show processing indicator
    await expect(page.getByText('AI is thinking')).toBeVisible();

    // Should eventually show AI response
    await expect(page.getByText('Thank you for sharing that!')).toBeVisible({ timeout: 2000 });
    await expect(page.getByText('AI Assistant')).toBeVisible();
  });

  test('should show conversation progress', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Should show progress bar
    const progressBar = page.locator('.bg-primary').first();
    await expect(progressBar).toBeVisible();

    // Should show current step
    await expect(page.getByText('Step: Initial')).toBeVisible();
  });

  test('should handle human validation prompts', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Mock human validation request
    await page.evaluate(() => {
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'human_validation_request',
            payload: {
              id: 'validation-123',
              conversation_id: 'conv-123',
              prompt: {
                id: 'validation-123',
                type: 'approval',
                question: 'Should we focus on iOS or Android first?',
                context: 'Based on your requirements, I recommend starting with iOS.',
                required: true
              }
            }
          }
        }));
      }, 1000);
    });

    // Should show validation alert
    await expect(page.getByText('Human Input Required')).toBeVisible({ timeout: 2000 });
    await expect(page.getByText('Should we focus on iOS or Android first?')).toBeVisible();

    // Should show validation buttons
    await expect(page.getByRole('button', { name: 'Approve' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reject' })).toBeVisible();

    // Click approve
    await page.getByRole('button', { name: 'Approve' }).click();

    // Should show confirmation
    await expect(page.getByText('✅ Approved')).toBeVisible();
  });

  test('should show and manage extracted requirements', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Send message that should trigger requirement extraction
    const messageInput = page.getByPlaceholder('Describe your project requirements');
    await messageInput.fill('The app must have user authentication and needs to sync with fitness trackers');
    await messageInput.press('Enter');

    // Mock requirement extraction
    await page.evaluate(() => {
      // Simulate requirement extraction
      setTimeout(() => {
        const event = new CustomEvent('requirements-extracted', {
          detail: {
            requirements: [
              'The app must have user authentication',
              'The app needs to sync with fitness trackers'
            ]
          }
        });
        window.dispatchEvent(event);
      }, 1000);
    });

    // Should show requirements button
    await expect(page.getByText('Requirements (2)')).toBeVisible({ timeout: 2000 });

    // Click to open requirements panel
    await page.getByText('Requirements (2)').click();

    // Should show extracted requirements
    await expect(page.getByText('Extracted Requirements')).toBeVisible();
    await expect(page.getByText('The app must have user authentication')).toBeVisible();
    await expect(page.getByText('The app needs to sync with fitness trackers')).toBeVisible();
  });

  test('should complete conversation and generate PRD', async ({ page }) => {
    // Mock PRD generation API
    await page.route('**/api/projects/*/prd/from-conversation', async route => {
      await route.fulfill({
        json: {
          data: {
            id: 'prd-123',
            title: 'Fitness Tracker App PRD',
            status: 'draft'
          }
        }
      });
    });

    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Simulate completed conversation state
    await page.evaluate(() => {
      // Mock conversation state to show completion
      const store = (window as any).$pinia?.state?.value?.conversation;
      if (store) {
        store.context.currentStep = 'completion';
        store.context.extractedRequirements = [
          'User authentication required',
          'Fitness tracker integration needed',
          'Real-time data sync'
        ];
      }
    });

    // Should show generate PRD button
    await expect(page.getByRole('button', { name: 'Generate PRD' })).toBeVisible();

    // Click generate PRD
    await page.getByRole('button', { name: 'Generate PRD' }).click();

    // Should show success message
    await expect(page.getByText('PRD generated successfully')).toBeVisible({ timeout: 3000 });
  });

  test('should handle conversation reset', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Send a message first
    const messageInput = page.getByPlaceholder('Describe your project requirements');
    await messageInput.fill('Test message');
    await messageInput.press('Enter');

    // Should show user message
    await expect(page.getByText('Test message')).toBeVisible();

    // Mock confirmation dialog
    page.on('dialog', dialog => {
      expect(dialog.message()).toContain('reset the conversation');
      dialog.accept();
    });

    // Click reset button
    await page.getByRole('button', { name: 'Reset' }).click();

    // Should show welcome message again
    await expect(page.getByText('Welcome! I\'ll help you create')).toBeVisible();
    
    // Previous message should be gone
    await expect(page.getByText('Test message')).not.toBeVisible();
  });

  test('should show connection status', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Should show connection status indicator
    const connectionStatus = page.locator('.w-2.h-2.rounded-full');
    await expect(connectionStatus).toBeVisible();

    // Mock disconnection
    await page.evaluate(() => {
      window.dispatchEvent(new CustomEvent('websocket-disconnect'));
    });

    // Should show disconnected status
    await expect(page.getByText('Disconnected')).toBeVisible();
  });

  test('should handle message regeneration', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Wait for welcome message and hover to show actions
    const assistantMessage = page.locator('.message-assistant').first();
    await assistantMessage.hover();

    // Should show regenerate button
    await expect(page.getByRole('button', { name: 'Regenerate' })).toBeVisible();

    // Click regenerate
    await page.getByRole('button', { name: 'Regenerate' }).click();

    // Should show processing indicator
    await expect(page.getByText('AI is thinking')).toBeVisible();
  });

  test('should copy message content', async ({ page }) => {
    // Grant clipboard permissions
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);

    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Wait for welcome message and hover to show actions
    const assistantMessage = page.locator('.message-assistant').first();
    await assistantMessage.hover();

    // Click copy button
    await page.getByRole('button', { name: 'Copy' }).click();

    // Should show success toast
    await expect(page.getByText('Copied')).toBeVisible({ timeout: 1000 });

    // Verify clipboard content
    const clipboardContent = await page.evaluate(() => navigator.clipboard.readText());
    expect(clipboardContent).toContain('Welcome');
  });

  test('should provide message feedback', async ({ page }) => {
    // Select project
    await page.getByText(TEST_PROJECT.name).click();

    // Wait for welcome message and hover to show actions
    const assistantMessage = page.locator('.message-assistant').first();
    await assistantMessage.hover();

    // Should show thumbs up/down buttons
    const thumbsUpButton = page.getByRole('button').filter({ hasText: /thumbs-up/ });
    const thumbsDownButton = page.getByRole('button').filter({ hasText: /thumbs-down/ });

    await expect(thumbsUpButton).toBeVisible();
    await expect(thumbsDownButton).toBeVisible();

    // Click thumbs up
    await thumbsUpButton.click();

    // Button should change color to indicate selection
    await expect(thumbsUpButton).toHaveClass(/text-green/);
  });
});

test.describe('Validation Prompt Component', () => {
  test.beforeEach(async ({ page }) => {
    // Setup authentication and navigate
    await page.route('**/api/auth/me', async route => {
      await route.fulfill({
        json: { id: 'user-123', email: 'testexample.com', name: 'Test User' }
      });
    });

    await page.route('**/api/projects', async route => {
      await route.fulfill({
        json: { data: [TEST_PROJECT] }
      });
    });

    await page.goto('/conversation');
    await page.getByText(TEST_PROJECT.name).click();
  });

  test('should handle approval validation', async ({ page }) => {
    // Mock approval validation request
    await page.evaluate(() => {
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'human_validation_request',
            payload: {
              prompt: {
                id: 'approval-test',
                type: 'approval',
                question: 'Do you approve this approach?',
                context: 'We suggest using React Native for cross-platform development.',
                required: true
              }
            }
          }
        }));
      }, 500);
    });

    // Should show validation prompt
    await expect(page.getByText('Do you approve this approach?')).toBeVisible({ timeout: 1000 });
    await expect(page.getByText('We suggest using React Native')).toBeVisible();

    // Should have approve and reject buttons
    const approveBtn = page.getByRole('button', { name: 'Approve' });
    const rejectBtn = page.getByRole('button', { name: 'Reject' });
    
    await expect(approveBtn).toBeVisible();
    await expect(rejectBtn).toBeVisible();

    // Test approval
    await approveBtn.click();
    await expect(page.getByText('✅ Approved')).toBeVisible();
  });

  test('should handle choice validation', async ({ page }) => {
    // Mock choice validation request
    await page.evaluate(() => {
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'human_validation_request',
            payload: {
              prompt: {
                id: 'choice-test',
                type: 'choice',
                question: 'Which platform should we target first?',
                context: 'We need to choose a primary platform for initial development.',
                options: [
                  { label: 'iOS', value: 'ios', description: 'Apple App Store' },
                  { label: 'Android', value: 'android', description: 'Google Play Store' },
                  { label: 'Web', value: 'web', description: 'Progressive Web App' }
                ],
                required: true
              }
            }
          }
        }));
      }, 500);
    });

    // Should show choice validation
    await expect(page.getByText('Which platform should we target first?')).toBeVisible({ timeout: 1000 });
    
    // Should show radio options
    await expect(page.getByText('iOS')).toBeVisible();
    await expect(page.getByText('Android')).toBeVisible();
    await expect(page.getByText('Web')).toBeVisible();

    // Select an option
    await page.getByRole('radio', { name: 'iOS Apple App Store' }).click();
    
    // Submit choice
    await page.getByRole('button', { name: 'Submit Choice' }).click();
    
    // Should show confirmation
    await expect(page.getByText('Choice validation completed')).toBeVisible();
  });

  test('should handle input validation', async ({ page }) => {
    // Mock input validation request
    await page.evaluate(() => {
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'human_validation_request',
            payload: {
              prompt: {
                id: 'input-test',
                type: 'input',
                question: 'Please specify your budget range?',
                context: 'We need budget information to recommend appropriate solutions.',
                required: true
              }
            }
          }
        }));
      }, 500);
    });

    // Should show input validation
    await expect(page.getByText('Please specify your budget range?')).toBeVisible({ timeout: 1000 });
    
    // Should show textarea
    const inputField = page.getByPlaceholder('Please provide your input...');
    await expect(inputField).toBeVisible();

    // Enter input
    await inputField.fill('$10,000 - $50,000 for the initial version');
    
    // Submit input
    await page.getByRole('button', { name: 'Submit' }).click();
    
    // Should show confirmation
    await expect(page.getByText('Input validation completed')).toBeVisible();
  });

  test('should show timeout countdown', async ({ page }) => {
    // Mock validation with timeout
    await page.evaluate(() => {
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: {
            type: 'human_validation_request',
            payload: {
              prompt: {
                id: 'timeout-test',
                type: 'approval',
                question: 'Quick decision needed!',
                context: 'This validation will timeout in 10 seconds.',
                timeout: 10000, // 10 seconds
                required: true
              }
            }
          }
        }));
      }, 500);
    });

    // Should show timeout indicator
    await expect(page.getByText('Time remaining:')).toBeVisible({ timeout: 1000 });
    await expect(page.getByText('10s')).toBeVisible();

    // Progress bar should be visible
    const progressBar = page.locator('.bg-amber-500');
    await expect(progressBar).toBeVisible();

    // Wait a bit and check if countdown decreases
    await page.waitForTimeout(2000);
    await expect(page.getByText('8s')).toBeVisible();
  });
});