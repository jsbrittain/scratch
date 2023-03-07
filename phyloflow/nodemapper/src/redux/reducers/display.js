import { createReducer } from "@reduxjs/toolkit"
import * as act from "../actions"

const displayStateInit = {
  show_settings_panel: false
};

// Display
export const displayReducer = createReducer(displayStateInit, {
  [act.displayOpenSettings]: (state, action) => {
	state.show_settings_panel = true;
	console.info("[Reducer] (display)OpenSettings");
  },
  [act.displayCloseSettings]: (state, action) => {
	state.show_settings_panel = false;
	console.info("[Reducer] (display)CloseSettings");
  },
  [act.displayToggleSettingsVisibility]: (state, action) => {
    state.show_settings_panel = !state.display.show_settings_panel;
	console.info("[Reducer] (display)ToggleSettingsVisibility");
  },

});
