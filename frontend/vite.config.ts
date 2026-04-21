import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    proxy: {
      // 将所有以 /api 开头的请求代理到后端 8000 端口
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // 如果后端接口没有 /api 前缀，可以重写路径
        // rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
});
