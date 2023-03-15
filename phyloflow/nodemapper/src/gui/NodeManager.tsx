import React from 'react'
import NodeMapEngine from './NodeMapEngine'
import { BodyWidget } from './BodyWidget'
import { nodemapNodeSelected } from '../redux/actions'
import { nodemapNodeDeselected } from '../redux/actions'
import { useAppSelector } from '../redux/store/hooks'
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

  // POST request handler [refactor out of this function later]
  const query = useAppSelector(state => state.nodemap.query);
  const [responseData, setResponseData] = React.useState('')
    async function getData() {
      const postRequestOptions = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json;charset=UTF-8',
        },
        body: JSON.stringify({
          query: 'tokenize',
          variables: {
            format: 'Snakefile',
            content: query
          }
        })
      }
    fetch('http://127.0.0.1:3001/tokenize', postRequestOptions)
      .then(response => {
        if (response.ok) {
          return response.json()
        }
        throw response
      })
      .then(data => {
        setResponseData(data);
        console.info("Got response: ", data); 
      })
      .catch(error => {
        console.error("Error during query: ", error);
      })
  }
  React.useEffect(() => {
    getData()
  }, [query]);

  return (
    <div id="nodemanager" style={{width: '100%', height: '100%'}}>
    <BodyWidget engine={engine} />
    </div>
  )
}

export default NodeManager
