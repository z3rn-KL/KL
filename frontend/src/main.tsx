import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import App from "./App";
import "./app.css";
import "./styles.css";
import "./styles/nexus-redesign.css";
import "./styles/precision-control-tower.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <TooltipProvider delay={200}>
        <App />
        <Toaster richColors position="bottom-right" />
      </TooltipProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
