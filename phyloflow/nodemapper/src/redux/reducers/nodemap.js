import { createReducer } from "@reduxjs/toolkit"
import * as act from "../actions"

// State
const nodemapStateInit = {
};

// Nodemap
export const nodemapReducer = createReducer(nodemapStateInit, {
  [act.nodemapAddNode]: (state, action) => {
    // Business logic
	console.info("[Reducer] (nodemap)AddNode");
  },
  [act.nodemapNodeSelected]: (state, action) => {
    // Action intercepted in middleware to control display
	console.info("[Reducer] (nodemap)NodeSelected");
  },
  [act.nodemapNodeDeselected]: (state, action) => {
    // Action intercepted in middleware to control display
	console.info("[Reducer] (nodemap)NodeDeselected");
  },
  [act.nodemapSelectNone]: (state, action) => {
	// Business logic
	console.info("[Reducer] (nodemap)SelectNone");
  },
});
