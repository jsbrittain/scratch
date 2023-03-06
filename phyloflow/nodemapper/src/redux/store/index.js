import { configureStore, getDefaultMiddleware } from "@reduxjs/toolkit"
import { nodemapReducer, displayReducer } from "../reducers"
import { sampleMiddleware } from "../middleware"


// Middleware layer ===============================================
const middleware = [
  ...getDefaultMiddleware(),
  // Custom middlewares go here...
  sampleMiddleware
];

// Reducers =======================================================
const store = configureStore({
  reducer: {
	nodemap: nodemapReducer,
	display: displayReducer
  },
  middleware
});

export default store;