import React from 'react'
import NodeMapEngine from './NodeMapEngine'
import { BodyWidget } from './BodyWidget'
import { nodemapNodeSelected } from '../redux/actions'
import { nodemapNodeDeselected } from '../redux/actions'
import { useAppDispatch } from '../redux/store/hooks'
import { DiagramModel } from "@projectstorm/react-diagrams"

import './NodeManager.css'

function NodeManager() {
  // Link to singleton instance of nodemap graph engine
  const nodeMapEngine = NodeMapEngine.Instance;
  const engine = nodeMapEngine.engine;
  const model = engine.getModel(); 
  // Add listeners, noting the following useful resource:
  // https://github.com/projectstorm/react-diagrams/issues/164
  const dispatch = useAppDispatch();
  model.getNodes().forEach(node =>
    node.registerListener({
      selectionChanged: (e) => {
        if (e.isSelected) {
          const payload = {
            id: node.options.id,
          }
          dispatch(nodemapNodeSelected(payload))
        }
      }
    })
  );
  return (
    <div id="nodemanager" style={{width: '100%', height: '100%'}}>
    <BodyWidget engine={engine} />
    </div>
  )
}

export default NodeManager
