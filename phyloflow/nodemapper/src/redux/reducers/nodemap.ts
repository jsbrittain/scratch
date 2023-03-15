import { createReducer } from "@reduxjs/toolkit"
import * as action from "../actions"

// State
const nodemapStateInit = {
  query: ''  // temp location
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
	  .addCase(action.nodemapConstructFromOutline, (state, action) => {
	    const payload = action.payload
	    console.info("[Reducer] (nodemap)ConstructFromOutline: ", payload);
      })
	  .addCase(action.nodemapSubmitQuery, (state, action) => {
	    const payload = action.payload
		state.query = payload
	    console.info("[Reducer] (nodemap)SubmitQuery: ", payload);
      })
  }
);

export default nodemapReducer
