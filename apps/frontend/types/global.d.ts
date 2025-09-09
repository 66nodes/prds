// Global type declarations for frontend
declare module 'tailwindcss' {
  export interface Config {
    content: string[];
    theme: {
      extend?: Record<string, any>;
      colors?: Record<string, any>;
    };
  }
}

declare module 'tailwindcss/defaultTheme' {
  const defaultTheme: Record<string, any>;
  export = defaultTheme;
}

declare module '@nuxt/schema' {
  export interface NuxtConfig {
    [key: string]: any;
  }
}

declare module '@nuxt/types' {
  export type Context = Record<string, any>;
  export type NuxtRuntimeConfig = Record<string, any>;
}

declare module '@tailwindcss/forms' {
  const plugin: any;
  export default plugin;
}

// Pinia persistence plugin types
declare module 'pinia-plugin-persist' {
  import { PiniaPlugin } from 'pinia';
  const piniaPersistPlugin: PiniaPlugin;
  export default piniaPersistPlugin;
}