import { createReducer } from "@reduxjs/toolkit"
import * as action from "../actions"

// State
const nodemapStateInit = {
};

// Nodemap
const nodemapReducer = createReducer(
  nodemapStateInit,
  (builder) => {
    builder
      .addCase(action.nodemapAddNode, (state, action) => {
        // Business logic
	    console.info("[Reducer] (nodemap)AddNode");
      })
	  .addCase(action.nodemapNodeSelected, (state, action) => {
        // Action intercepted in middleware to control display
	    console.info("[Reducer] (nodemap)NodeSelected", state, action);
  	    //var config = JSON.parse(node.options.extras);
      })
	  .addCase(action.nodemapNodeDeselected, (state, action) => {
        // Action intercepted in middleware to control display
	    console.info("[Reducer] (nodemap)NodeDeselected");
      })
	  .addCase(action.nodemapSelectNone, (state, action) => {
	    // Business logic
	    console.info("[Reducer] (nodemap)SelectNone");
      })
  }
);

export default nodemapReducer
