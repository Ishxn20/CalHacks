import { ConvexProvider, ConvexReactClient } from "convex/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const url = import.meta.env.VITE_CONVEX_URL as string | undefined;
const root = createRoot(document.getElementById("root")!);
if (!url) {
  root.render(<p className="setup">Set VITE_CONVEX_URL (run <code>npx convex dev</code> in fleet/) and reload.</p>);
} else {
  root.render(
    <StrictMode>
      <ConvexProvider client={new ConvexReactClient(url)}>
        <App />
      </ConvexProvider>
    </StrictMode>,
  );
}
