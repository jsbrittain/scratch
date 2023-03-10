import { displayOpenSettings, displayCloseSettings, displayUpdateCodeSnippet } from '../actions'
import NodeMapEngine from '../../gui/NodeMapEngine' 

export function nodemapMiddleware({ getState, dispatch }) {
  return function(next) {
	return function(action) {
	  // action.type
	  //       .payload
	  console.log("Middleware: ", action );
	  switch (action.type) {
          case "nodemap/node-selected": {
			if (NodeMapEngine.Instance.lock) {
			  const nodemap = NodeMapEngine.Instance;
			  const node = nodemap.getNodeById(action.payload.id);
			  let payload = {}
			  if (node === null) {
				console.log("NODE NOT FOUND")
			    payload = action.payload.id
			  } else {
			    const json = JSON.parse(node.options.extras);
			    payload = {
			      id: action.payload.id,
			  	  name: node.options.name,
				  codesnippet: json.codesnippet,
			    }
			  }
			  dispatch(displayUpdateCodeSnippet(JSON.stringify(payload)));
			  dispatch(displayOpenSettings());
			}
			break;
		  }
		  case "nodemap/node-deselected": {
			dispatch(displayCloseSettings());
			break;
		  }
		  case "nodemap/select-none": {
			// Link to singleton instance of nodemap graph engine
			const nodemap = NodeMapEngine.Instance;
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
