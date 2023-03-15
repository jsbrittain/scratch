import { createAction } from "@reduxjs/toolkit"

export const nodemapAddNode = createAction("nodemap/add-node");
export const nodemapViewSettings = createAction("nodemap/view-settings");
export const nodemapNodeSelected = createAction<Record<string, any> | undefined>("nodemap/node-selected");
export const nodemapNodeDeselected = createAction("nodemap/node-deselected");
export const nodemapSelectNone = createAction("nodemap/select-none");
export const nodemapConstructFromOutline = createAction("nodemap/construct-from-outline")

// refactor out later
export const nodemapSubmitQuery = createAction<string>("nodemap/submit-query")
