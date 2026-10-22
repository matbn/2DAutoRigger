import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./editor.css";

const container = document.getElementById("root");
if (!container) {
  throw new Error("no #root element");
}

createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
