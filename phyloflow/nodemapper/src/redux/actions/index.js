import { createAction } from "@reduxjs/toolkit"

// Actions

export const nodemapAddNode = createAction("NODEMAP_ADD_NODE");
export const nodemapViewSettings = createAction("NODEMAP_VIEW_SETTINGS");
export const nodemapInitializeEngine = createAction("NODEMAP_INIT_ENGINE");

export const displayOpenSettings = createAction("DISPLAY_OPEN_SETTINGS");
export const displayCloseSettings = createAction("DISPLAY_CLOSE_SETTINGS");
export const displayToggleSettingsVisibility = createAction("DISPLAY_TOGGLE_SETTINGS_VISIBILITY");
