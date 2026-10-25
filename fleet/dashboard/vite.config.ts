import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// .env.local (written by `npx convex dev`) lives one level up, in fleet/.
export default defineConfig({ plugins: [react()], envDir: ".." });
