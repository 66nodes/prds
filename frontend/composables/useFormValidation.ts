import { ref, computed, watch } from 'vue'
import type { Ref } from 'vue'

export interface ValidationRule {
  validate: (value: any) => boolean | string
  message?: string
}

export interface FieldValidation {
  value: Ref<any>
  rules: ValidationRule[]
  error: Ref<string | null>
  touched: Ref<boolean>
  dirty: Ref<boolean>
}

export interface FormValidation {
  fields: Map<string, FieldValidation>
  isValid: Ref<boolean>
  isDirty: Ref<boolean>
  errors: Ref<Record<string, string | null>>
  validate: () => boolean
  validateField: (name: string) => boolean
  reset: () => void
  setFieldError: (name: string, error: string) => void
  clearFieldError: (name: string) => void
}

// Common validation rules
export const validators = {
  required: (message = 'This field is required'): ValidationRule => ({
    validate: (value) => {
      if (Array.isArray(value)) return value.length > 0
      if (typeof value === 'string') return value.trim().length > 0
      return value != null && value !== ''
    },
    message
  }),
  
  email: (message = 'Please enter a valid email'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true // Let required handle empty values
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      return emailRegex.test(value)
    },
    message
  }),
  
  minLength: (min: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      return value.length >= min
    },
    message: message || `Must be at least ${min} characters`
  }),
  
  maxLength: (max: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      return value.length <= max
    },
    message: message || `Must be no more than ${max} characters`
  }),
  
  pattern: (regex: RegExp, message = 'Invalid format'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      return regex.test(value)
    },
    message
  }),
  
  url: (message = 'Please enter a valid URL'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      try {
        new URL(value)
        return true
      } catch {
        return false
      }
    },
    message
  }),
  
  number: (message = 'Must be a number'): ValidationRule => ({
    validate: (value) => {
      if (!value) return true
      return !isNaN(Number(value))
    },
    message
  }),
  
  min: (min: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (value === null || value === undefined || value === '') return true
      return Number(value) >= min
    },
    message: message || `Must be at least ${min}`
  }),
  
  max: (max: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (value === null || value === undefined || value === '') return true
      return Number(value) <= max
    },
    message: message || `Must be no more than ${max}`
  }),
  
  arrayMinLength: (min: number, message?: string): ValidationRule => ({
    validate: (value) => {
      if (!Array.isArray(value)) return false
      return value.filter(v => v !== '' && v != null).length >= min
    },
    message: message || `At least ${min} item(s) required`
  }),
  
  custom: (fn: (value: any) => boolean | string, message?: string): ValidationRule => ({
    validate: fn,
    message
  })
}

export const useFormValidation = (
  initialFields: Record<string, { value: Ref<any>; rules: ValidationRule[] }>
): FormValidation => {
  const fields = new Map<string, FieldValidation>()
  
  // Initialize fields
  Object.entries(initialFields).forEach(([name, config]) => {
    fields.set(name, {
      value: config.value,
      rules: config.rules,
      error: ref<string | null>(null),
      touched: ref(false),
      dirty: ref(false)
    })
  })
  
  // Computed errors object
  const errors = computed(() => {
    const errorObj: Record<string, string | null> = {}
    fields.forEach((field, name) => {
      errorObj[name] = field.error.value
    })
    return errorObj
  })
  
  // Computed validation state
  const isValid = computed(() => {
    for (const [_, field] of fields) {
      if (field.error.value) return false
    }
    return true
  })
  
  // Computed dirty state
  const isDirty = computed(() => {
    for (const [_, field] of fields) {
      if (field.dirty.value) return true
    }
    return false
  })
  
  // Validate a single field
  const validateField = (name: string): boolean => {
    const field = fields.get(name)
    if (!field) return true
    
    field.touched.value = true
    field.error.value = null
    
    for (const rule of field.rules) {
      const result = rule.validate(field.value.value)
      if (result !== true) {
        field.error.value = typeof result === 'string' ? result : rule.message || 'Validation failed'
        return false
      }
    }
    
    return true
  }
  
  // Validate all fields
  const validate = (): boolean => {
    let allValid = true
    fields.forEach((_, name) => {
      if (!validateField(name)) {
        allValid = false
      }
    })
    return allValid
  }
  
  // Set field error manually
  const setFieldError = (name: string, error: string) => {
    const field = fields.get(name)
    if (field) {
      field.error.value = error
      field.touched.value = true
    }
  }
  
  // Clear field error
  const clearFieldError = (name: string) => {
    const field = fields.get(name)
    if (field) {
      field.error.value = null
    }
  }
  
  // Reset form
  const reset = () => {
    fields.forEach((field) => {
      field.error.value = null
      field.touched.value = false
      field.dirty.value = false
    })
  }
  
  // Watch for changes and mark as dirty
  fields.forEach((field, name) => {
    watch(field.value, () => {
      field.dirty.value = true
      // Auto-validate on change if field was touched
      if (field.touched.value) {
        validateField(name)
      }
    })
  })
  
  return {
    fields,
    isValid,
    isDirty,
    errors,
    validate,
    validateField,
    reset,
    setFieldError,
    clearFieldError
  }
}