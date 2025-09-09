module.exports = {
  extends: ['plugin:vue/vue3-recommended', 'plugin:vue/vue3-essential'],
  plugins: ['vue'],
  rules: {
    'vue/multi-word-component-names': 'off',
    'vue/no-v-html': 'warn',
    'vue/require-default-prop': 'warn',
    'vue/require-prop-types': 'warn',
  },
};
