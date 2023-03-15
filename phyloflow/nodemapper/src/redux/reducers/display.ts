import { createReducer } from "@reduxjs/toolkit"
import * as action from "../actions"

const displayStateInit = {
  show_settings_panel: false,
  settings_title: "",
  codesnippet: "",
};

// Display
const displayReducer = createReducer(
  displayStateInit,
  (builder) => {
    builder
	  .addCase(action.displayOpenSettings, (state, action) => {
	    state.show_settings_panel = true;
	    console.info("[Reducer] (display)OpenSettings", state, action);
      })
      .addCase(action.displayCloseSettings, (state, action) => {
        state.show_settings_panel = false;
	    console.info("[Reducer] (display)CloseSettings", state, action);
      })
      .addCase(action.displayToggleSettingsVisibility, (state, action) => {
        state.show_settings_panel = !state.show_settings_panel;
	    console.info("[Reducer] (display)ToggleSettingsVisibility", state, action);
      })
      .addCase(action.displayUpdateCodeSnippet, (state, action) => {
		state.settings_title = action.payload;
        state.codesnippet = action.payload;
	    console.info("[Reducer] (display)UpdateCodeSnippet", state, action);
      })
  }
);

export default displayReducer
