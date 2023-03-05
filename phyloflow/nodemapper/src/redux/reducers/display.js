import { createReducer } from "@reduxjs/toolkit"
import * as act from "../actions"

const displayStateInit = {
  show_settings_panel: false
};

// Display
export const displayReducer = createReducer(displayStateInit, {
  [act.displayOpenSettings]: (state, action) => {
	  state.show_settings_panel = true;
  },
  [act.displayCloseSettings]: (state, action) => {
	  state.show_settings_panel = false;
  },
  [act.displayToggleSettingsVisibility]: (state, action) => {
    state.show_settings_panel = !state.display.show_settings_panel;
  },

});
