import { displayOpenSettings, displayCloseSettings } from '../actions'
import NodeMapEngine from '../../gui/NodeMapEngine' 

export function nodemapMiddleware({ getState, dispatch }) {
  return function(next) {
	return function(action) {
	  // action.type
	  //       .payload
	  console.debug(action);
	  switch (action.type) {
		  case "NODEMAP_NODE_SELECTED": {
			if (NodeMapEngine.Instance.lock) {
			  dispatch(displayOpenSettings());
			}
			break;
		  }
		  case "NODEMAP_NODE_DESELECTED": {
			dispatch(displayCloseSettings());
			break;
		  }
		  case "NODEMAP_SELECT_NONE": {
			// Link to singleton instance of nodemap graph engine
			let nodemap = NodeMapEngine.Instance;
			nodemap.NodesSelectNone();
            break;
		  }
		  default:
			break;
	  }
	  
	  return next(action)
	}
  }
}
