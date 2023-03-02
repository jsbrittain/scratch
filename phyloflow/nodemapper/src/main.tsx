import * as React from 'react';
import { StrictMode } from 'react';
import { createRoot } from "react-dom/client";
import "./main.css";

import App from "./PanelManager";

const root = createRoot(document.getElementById("app"));
root.render(<StrictMode><App /></StrictMode>);
