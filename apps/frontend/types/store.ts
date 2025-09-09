// Store type declarations
export interface StoreInterface {
  get: (key: string) => any;
  set: (key: string, value: any) => void;
  remove: (key: string) => void;
  clearAll: () => void;
  clearNamespace: (namespace: string) => void;
  getOrSet: (key: string, computeFn: () => any) => any;
  getStats: () => {
    hits: number;
    misses: number;
    evictions: number;
    totalRequests: number;
  };
}

export interface CacheStore extends StoreInterface {
  _buildCacheKey: (key: string) => string;
  _isExpired: (entry: any) => boolean;
  _evictOldest: () => void;
  getOrSet: (key: string, computeFn: () => any, ttl?: number) => any;
  remove: (key: string) => boolean;
  clearNamespace: (namespace: string) => boolean;
  getStats: () => {
    hits: number;
    misses: number;
    evictions: number;
    totalRequests: number;
    hitRate: number;
    cacheSize: number;
    storageUsage: number;
  };
  hitRate: number;
  cacheSize: number;
  storageUsage: number;
  getLLMResponse: (key: string) => any;
  cacheLLMResponse: (key: string, value: any) => void;
}