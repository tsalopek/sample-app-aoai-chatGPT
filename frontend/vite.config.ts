import react from '@vitejs/plugin-react'
import { execFileSync } from 'node:child_process'
import { defineConfig } from 'vite'

const getBuildId = (): string => {
  if (process.env.BUILD_ID) return process.env.BUILD_ID

  try {
    return execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], {
      encoding: 'utf8'
    }).trim()
  } catch {
    return 'unknown'
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __BUILD_ID__: JSON.stringify(getBuildId())
  },
  build: {
    outDir: '../static',
    emptyOutDir: true,
    sourcemap: true
  },
  server: {
    proxy: {
      '/ask': 'http://localhost:5000',
      '/chat': 'http://localhost:5000'
    }
  }
})
