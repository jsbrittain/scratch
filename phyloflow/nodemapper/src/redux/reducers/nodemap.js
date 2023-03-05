import { createReducer } from "@reduxjs/toolkit"
import * as act from "../actions"

// State
const nodemapStateInit = {
};

// Nodemap
export const nodemapReducer = createReducer(nodemapStateInit, {
  [act.nodemapAddNode]: (state, action) => {
    // Business logic
  },
  [act.nodemapNodeSelected]: (state, action) => {
    // Action intercepted in middleware to control display
  },
  [act.nodemapNodeDeselected]: (state, action) => {
    // Action intercepted in middleware to control display
  },
});
