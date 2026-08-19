import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/search': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/assets': 'http://127.0.0.1:8000',
      '/videos': 'http://127.0.0.1:8000'
    }
  }
});
