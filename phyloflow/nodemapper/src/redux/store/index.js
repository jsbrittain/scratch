import { configureStore, getDefaultMiddleware } from "@reduxjs/toolkit"
import { nodemapReducer, displayReducer } from "../reducers"
import { sampleMiddleware } from "../middleware"

const middleware = [
  ...getDefaultMiddleware(),
  // Custom middlewares go here...
  sampleMiddleware
];

const store = configureStore({
  reducer: {
	nodemap: nodemapReducer,
	display: displayReducer
  },
  middleware
});

export default store;
