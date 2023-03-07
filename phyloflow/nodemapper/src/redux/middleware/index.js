import { displayOpenSettings, displayCloseSettings, nodemapSelectNone } from '../actions'

export function sampleMiddleware({ getState, dispatch }) {
  return function(next) {
	return function(action) {
	  
	  switch (action.type) {
		  case "NODEMAP_NODE_SELECTED": {
			dispatch(displayOpenSettings());
			break;
		  }
		  case "NODEMAP_NODE_DESELECTED": {
			dispatch(displayCloseSettings());
			break;
		  }
		  case "DISPLAY_CLOSE_SETTINGS": {
			dispatch(nodemapSelectNone());
			break;
		  }
		  default:
			break;
	  }
	  
	  return next(action)
	}
  }
}
