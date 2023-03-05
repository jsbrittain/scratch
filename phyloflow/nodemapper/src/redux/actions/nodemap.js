import { createAction } from "@reduxjs/toolkit"

export const nodemapAddNode = createAction("NODEMAP_ADD_NODE");
export const nodemapViewSettings = createAction("NODEMAP_VIEW_SETTINGS");
export const nodemapNodeSelected = createAction("NODEMAP_NODE_SELECTED");
export const nodemapNodeDeselected = createAction("NODEMAP_NODE_DESELECTED");
