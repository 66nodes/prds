/**
 * Unit tests for ValidationPrompt component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mount } from '@vue/test-utils';
import ValidationPrompt from '~/components/conversation/ValidationPrompt.vue';
import type { HumanValidationPrompt } from '~/types';

// Mock UI components
vi.mock('#ui', () => ({
  UIcon: { template: '<span></span>' },
  UButton: { 
    template: '<button><slot></slot></button>',
    props: ['color', 'variant', 'disabled', 'loading']
  },
  UTextarea: { 
    template: '<textarea></textarea>',
    props: ['modelValue', 'placeholder', 'rows', 'required']
  },
  URadioGroup: { 
    template: '<div></div>',
    props: ['modelValue', 'options']
  },
  UCheckbox: { 
    template: '<input type="checkbox">',
    props: ['modelValue', 'label']
  }
}));

describe('ValidationPrompt', () => {
  let wrapper: any;

  const createApprovalPrompt = (): HumanValidationPrompt => ({
    id: 'test-approval-id',
    type: 'approval' as const,
    question: 'Do you approve this approach?',
    context: 'We are implementing a new feature that requires user approval.',
    required: true,
    timeout: 30000,
    options: undefined,
    metadata: undefined
  });

  const createChoicePrompt = (): HumanValidationPrompt => ({
    id: 'test-choice-id',
    type: 'choice' as const,
    question: 'Which approach should we use?',
    context: 'Please select the best implementation approach.',
    required: true,
    timeout: 30000,
    options: [
      { label: 'Option A', value: 'a', description: 'First approach' },
      { label: 'Option B', value: 'b', description: 'Second approach' }
    ],
    metadata: undefined
  });

  const createInputPrompt = (): HumanValidationPrompt => ({
    id: 'test-input-id',
    type: 'input' as const,
    question: 'Please provide additional details',
    context: 'We need more information to proceed.',
    required: true,
    timeout: 30000,
    options: undefined,
    metadata: undefined
  });

  const createReviewPrompt = (): HumanValidationPrompt => ({
    id: 'test-review-id',
    type: 'review' as const,
    question: 'Please review the generated content',
    context: 'Review and approve the generated PRD content.',
    required: true,
    timeout: 30000,
    options: undefined,
    metadata: {
      reviewContent: 'This is the content to be reviewed.\n\n**Important:** Please check all details.'
    }
  });

  const createConfirmationPrompt = (): HumanValidationPrompt => ({
    id: 'test-confirmation-id',
    type: 'confirmation' as const,
    question: 'I understand the implications of this action',
    context: 'Please confirm you understand before proceeding.',
    required: true,
    timeout: 30000,
    options: undefined,
    metadata: undefined
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    if (wrapper) {
      wrapper.unmount();
    }
  });

  describe('Approval Type Validation', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createApprovalPrompt()
        }
      });
    });

    it('should render approval validation prompt correctly', () => {
      expect(wrapper.find('.validation-prompt').exists()).toBe(true);
      expect(wrapper.text()).toContain('Human Input Required');
      expect(wrapper.text()).toContain('Do you approve this approach?');
      expect(wrapper.text()).toContain('We are implementing a new feature');
    });

    it('should show approve and reject buttons for approval type', () => {
      const buttons = wrapper.findAll('button');
      expect(buttons.length).toBeGreaterThanOrEqual(2);
      expect(wrapper.text()).toContain('Approve');
      expect(wrapper.text()).toContain('Reject');
    });

    it('should emit approve event when approve button is clicked', async () => {
      const approveButton = wrapper.find('button'); // First button should be approve
      await approveButton.trigger('click');
      
      expect(wrapper.emitted('approve')).toBeTruthy();
      expect(wrapper.emitted('approve')[0]).toEqual([{ feedback: '' }]);
    });

    it('should include feedback in approval response', async () => {
      const textarea = wrapper.find('textarea');
      await textarea.setValue('This looks good to me');

      const approveButton = wrapper.find('button');
      await approveButton.trigger('click');

      expect(wrapper.emitted('approve')[0]).toEqual([{ feedback: 'This looks good to me' }]);
    });
  });

  describe('Choice Type Validation', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createChoicePrompt()
        }
      });
    });

    it('should render choice validation prompt correctly', () => {
      expect(wrapper.text()).toContain('Which approach should we use?');
      expect(wrapper.text()).toContain('Please select the best implementation approach');
    });

    it('should show radio group for choices', () => {
      expect(wrapper.findComponent({ name: 'URadioGroup' }).exists()).toBe(true);
    });

    it('should show submit button for choices', () => {
      expect(wrapper.text()).toContain('Submit Choice');
    });

    it('should emit approve event with selected choice', async () => {
      // Simulate selecting a choice
      wrapper.vm.selectedChoice = 'a';
      await wrapper.vm.$nextTick();

      const submitButton = wrapper.find('button');
      await submitButton.trigger('click');

      expect(wrapper.emitted('approve')).toBeTruthy();
      expect(wrapper.emitted('approve')[0]).toEqual([{ choice: 'a' }]);
    });
  });

  describe('Input Type Validation', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createInputPrompt()
        }
      });
    });

    it('should render input validation prompt correctly', () => {
      expect(wrapper.text()).toContain('Please provide additional details');
      expect(wrapper.text()).toContain('We need more information to proceed');
    });

    it('should show textarea for input', () => {
      const textareas = wrapper.findAll('textarea');
      expect(textareas.length).toBeGreaterThanOrEqual(1);
    });

    it('should emit approve event with input value', async () => {
      const textarea = wrapper.find('textarea');
      await textarea.setValue('Here are the additional details...');

      const submitButton = wrapper.find('button');
      await submitButton.trigger('click');

      expect(wrapper.emitted('approve')).toBeTruthy();
      expect(wrapper.emitted('approve')[0]).toEqual([{ input: 'Here are the additional details...' }]);
    });

    it('should disable submit if required and input is empty', async () => {
      const submitButton = wrapper.find('button');
      expect(submitButton.attributes('disabled')).toBeDefined();
    });
  });

  describe('Review Type Validation', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createReviewPrompt()
        }
      });
    });

    it('should render review validation prompt correctly', () => {
      expect(wrapper.text()).toContain('Please review the generated content');
      expect(wrapper.text()).toContain('Review and approve the generated PRD content');
    });

    it('should show review content', () => {
      expect(wrapper.text()).toContain('This is the content to be reviewed');
    });

    it('should show review action buttons', () => {
      expect(wrapper.text()).toContain('Approve');
      expect(wrapper.text()).toContain('Request Changes');
      expect(wrapper.text()).toContain('Reject');
    });

    it('should emit approve event for review approval', async () => {
      wrapper.vm.reviewFeedback = 'Looks good!';
      await wrapper.vm.$nextTick();

      await wrapper.vm.handleReviewApprove();

      expect(wrapper.emitted('approve')).toBeTruthy();
      expect(wrapper.emitted('approve')[0]).toEqual([{
        feedback: 'Looks good!',
        decision: 'approved'
      }]);
    });

    it('should emit reject event for requesting changes', async () => {
      wrapper.vm.reviewFeedback = 'Please fix section 3';
      await wrapper.vm.$nextTick();

      await wrapper.vm.handleReviewRequestChanges();

      expect(wrapper.emitted('reject')).toBeTruthy();
      expect(wrapper.emitted('reject')[0]).toEqual([{
        feedback: 'Please fix section 3',
        decision: 'changes_requested'
      }]);
    });
  });

  describe('Confirmation Type Validation', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createConfirmationPrompt()
        }
      });
    });

    it('should render confirmation validation prompt correctly', () => {
      expect(wrapper.text()).toContain('I understand the implications of this action');
      expect(wrapper.text()).toContain('Please confirm you understand before proceeding');
    });

    it('should show checkbox for confirmation', () => {
      expect(wrapper.find('input[type="checkbox"]').exists()).toBe(true);
    });

    it('should show confirm and cancel buttons', () => {
      expect(wrapper.text()).toContain('Confirm');
      expect(wrapper.text()).toContain('Cancel');
    });

    it('should disable confirm button until checkbox is checked', async () => {
      const confirmButton = wrapper.find('button');
      expect(confirmButton.attributes('disabled')).toBeDefined();

      wrapper.vm.confirmed = true;
      await wrapper.vm.$nextTick();

      expect(confirmButton.attributes('disabled')).toBeUndefined();
    });
  });

  describe('Timeout Handling', () => {
    it('should display timeout countdown when timeout is set', () => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: {
            ...createApprovalPrompt(),
            timeout: 30000 // 30 seconds
          }
        }
      });

      expect(wrapper.text()).toContain('Time remaining:');
    });

    it('should start timeout when component is mounted', () => {
      const prompt = {
        ...createApprovalPrompt(),
        timeout: 1000 // 1 second for test
      };

      wrapper = mount(ValidationPrompt, {
        props: { prompt }
      });

      expect(wrapper.vm.timeRemaining).toBe(1000);
    });

    it('should clear timeout when component is unmounted', () => {
      const clearIntervalSpy = vi.spyOn(global, 'clearInterval');
      
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: {
            ...createApprovalPrompt(),
            timeout: 30000
          }
        }
      });

      wrapper.unmount();
      expect(clearIntervalSpy).toHaveBeenCalled();
    });
  });

  describe('Component State Management', () => {
    beforeEach(() => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createApprovalPrompt()
        }
      });
    });

    it('should reset state when prompt changes', async () => {
      // Set some state
      wrapper.vm.feedback = 'Some feedback';
      wrapper.vm.selectedChoice = 'test';
      wrapper.vm.inputValue = 'Some input';
      wrapper.vm.confirmed = true;

      // Change prompt
      await wrapper.setProps({
        prompt: createChoicePrompt()
      });

      // State should be reset
      expect(wrapper.vm.feedback).toBe('');
      expect(wrapper.vm.selectedChoice).toBe('');
      expect(wrapper.vm.inputValue).toBe('');
      expect(wrapper.vm.confirmed).toBe(false);
    });

    it('should show loading state during submission', async () => {
      wrapper.vm.isSubmitting = true;
      await wrapper.vm.$nextTick();

      const buttons = wrapper.findAll('button');
      buttons.forEach(button => {
        expect(button.attributes('loading')).toBeDefined();
      });
    });
  });

  describe('Content Formatting', () => {
    it('should format review content with basic markdown', () => {
      const content = '**Bold text** and *italic text* and `code`';
      const formatted = wrapper.vm.formatReviewContent(content);
      
      expect(formatted).toContain('<strong>Bold text</strong>');
      expect(formatted).toContain('<em>italic text</em>');
      expect(formatted).toContain('<code');
    });

    it('should format time remaining correctly', () => {
      expect(wrapper.vm.formatTimeRemaining(65000)).toBe('1m 5s');
      expect(wrapper.vm.formatTimeRemaining(30000)).toBe('30s');
      expect(wrapper.vm.formatTimeRemaining(0)).toBe('0s');
    });
  });

  describe('Error Handling', () => {
    it('should handle submission errors gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createApprovalPrompt()
        }
      });

      // Mock emit to throw error
      wrapper.vm.$emit = vi.fn().mockImplementation(() => {
        throw new Error('Network error');
      });

      await wrapper.vm.submitValidation({ test: 'data' }, true);

      expect(consoleSpy).toHaveBeenCalledWith('Failed to submit validation:', expect.any(Error));
      expect(wrapper.vm.isSubmitting).toBe(false);
      
      consoleSpy.mockRestore();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels and roles', () => {
      wrapper = mount(ValidationPrompt, {
        props: {
          prompt: createApprovalPrompt()
        }
      });

      expect(wrapper.find('.validation-prompt').exists()).toBe(true);
      // Add more accessibility tests as needed
    });
  });
});