import { createAction } from "@reduxjs/toolkit"

export const displayOpenSettings = createAction("display/open-settings");
export const displayCloseSettings = createAction("display/close-settings");
export const displayToggleSettingsVisibility = createAction("display/toggle-settings-visibility");
//export const displayUpdateCodeSnippet = createAction<Record<string,any>>("display/update-codesnippet")
export const displayUpdateCodeSnippet = createAction<string>("display/update-codesnippet")
