import type { ApiResponse, PaginatedResponse } from '~/types'

interface RequestOptions {
  headers?: Record<string, string>
  params?: Record<string, any>
  retry?: number
  timeout?: number
}

export const useApiClient = () => {
  const config = useRuntimeConfig()
  const { refreshAccessToken } = useAuth()
  
  // Get current access token
  const getAccessToken = (): string | null => {
    const accessToken = useCookie('access_token')
    return accessToken.value
  }

  // Build headers with authentication
  const buildHeaders = (customHeaders?: Record<string, string>): Record<string, string> => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...customHeaders
    }

    const token = getAccessToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    return headers
  }

  // Handle API errors
  const handleApiError = async (error: any, retry: boolean = true): Promise<void> => {
    // Handle 401 Unauthorized - try to refresh token
    if (error.status === 401 && retry) {
      const refreshed = await refreshAccessToken()
      if (refreshed) {
        // Token refreshed, caller should retry the request
        throw { retry: true }
      }
    }

    // Handle other errors
    const message = error?.data?.message || error?.message || 'An unexpected error occurred'
    const errors = error?.data?.errors || []
    
    throw {
      message,
      errors,
      status: error.status,
      data: error.data
    }
  }

  // GET request
  const get = async <T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> => {
    try {
      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'GET',
        headers: buildHeaders(options.headers),
        params: options.params,
        retry: options.retry || 0,
        timeout: options.timeout || 30000
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return get<T>(endpoint, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // POST request
  const post = async <T>(
    endpoint: string,
    body?: any,
    options: RequestOptions = {}
  ): Promise<T> => {
    try {
      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'POST',
        headers: buildHeaders(options.headers),
        body,
        params: options.params,
        retry: options.retry || 0,
        timeout: options.timeout || 30000
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return post<T>(endpoint, body, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // PUT request
  const put = async <T>(
    endpoint: string,
    body?: any,
    options: RequestOptions = {}
  ): Promise<T> => {
    try {
      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'PUT',
        headers: buildHeaders(options.headers),
        body,
        params: options.params,
        retry: options.retry || 0,
        timeout: options.timeout || 30000
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return put<T>(endpoint, body, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // PATCH request
  const patch = async <T>(
    endpoint: string,
    body?: any,
    options: RequestOptions = {}
  ): Promise<T> => {
    try {
      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'PATCH',
        headers: buildHeaders(options.headers),
        body,
        params: options.params,
        retry: options.retry || 0,
        timeout: options.timeout || 30000
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return patch<T>(endpoint, body, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // DELETE request
  const del = async <T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> => {
    try {
      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'DELETE',
        headers: buildHeaders(options.headers),
        params: options.params,
        retry: options.retry || 0,
        timeout: options.timeout || 30000
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return del<T>(endpoint, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // Get paginated data
  const getPaginated = async <T>(
    endpoint: string,
    page: number = 1,
    pageSize: number = 20,
    options: RequestOptions = {}
  ): Promise<PaginatedResponse<T>> => {
    try {
      const response = await $fetch<ApiResponse<PaginatedResponse<T>>>(
        `${config.public.apiBase}${endpoint}`,
        {
          method: 'GET',
          headers: buildHeaders(options.headers),
          params: {
            page,
            pageSize,
            ...options.params
          },
          retry: options.retry || 0,
          timeout: options.timeout || 30000
        }
      )

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return getPaginated<T>(endpoint, page, pageSize, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // Upload file
  const upload = async <T>(
    endpoint: string,
    file: File,
    additionalData?: Record<string, any>,
    options: RequestOptions = {}
  ): Promise<T> => {
    const formData = new FormData()
    formData.append('file', file)

    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value)
      })
    }

    try {
      const headers = buildHeaders(options.headers)
      delete headers['Content-Type'] // Let browser set multipart/form-data

      const response = await $fetch<ApiResponse<T>>(`${config.public.apiBase}${endpoint}`, {
        method: 'POST',
        headers,
        body: formData,
        retry: options.retry || 0,
        timeout: options.timeout || 60000 // Longer timeout for uploads
      })

      if (!response.success) {
        throw { data: response }
      }

      return response.data
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return upload<T>(endpoint, file, additionalData, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
      throw error
    }
  }

  // Download file
  const download = async (
    endpoint: string,
    filename: string,
    options: RequestOptions = {}
  ): Promise<void> => {
    try {
      const response = await $fetch.raw(`${config.public.apiBase}${endpoint}`, {
        method: 'GET',
        headers: buildHeaders(options.headers),
        params: options.params,
        responseType: 'blob',
        retry: options.retry || 0,
        timeout: options.timeout || 60000
      })

      // Create download link
      const blob = response._data as Blob
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error: any) {
      if (error?.retry) {
        // Retry with new token
        return download(endpoint, filename, { ...options, retry: 0 })
      }
      await handleApiError(error, options.retry !== 0)
    }
  }

  return {
    get,
    post,
    put,
    patch,
    delete: del,
    getPaginated,
    upload,
    download,
    $api: {
      get,
      post,
      put,
      patch,
      delete: del
    }
  }
}