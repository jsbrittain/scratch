import { createReducer } from "@reduxjs/toolkit"
import * as act from "../actions"
import InitializeEngine from '../../gui/SetupNodeScene'

// State structure
const nodemapStateInit = {
  engine: InitializeEngine()
};

const displayStateInit = {
  show_settings_panel: false
};

// Nodemap
export const nodemapReducer = createReducer(nodemapStateInit, {
  [act.nodemapAddNode]: (state, action) => {
    // Business logic
  },
  [act.nodemapInitializeEngine]: (state, action) => {
    state.nodemap.engine = InitializeEngine();
  },
});

// Display
export const displayReducer = createReducer(displayStateInit, {
  [act.displayOpenSettings]: (state, action) => {
	  state.display.show_settings_panel = true;
  },
  [act.displayCloseSettings]: (state, action) => {
	  state.display.show_settings_panel = false;
  },
  [act.displayToggleSettingsVisibility]: (state, action) => {
    state.display.show_settings_panel = !state.display.show_settings_panel;
  },

});
