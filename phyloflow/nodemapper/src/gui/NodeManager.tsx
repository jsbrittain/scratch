import React from 'react'
import { Component, StrictMode, useEffect } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'

import { BodyWidget } from './BodyWidget'
import './NodeManager.css'
import NodeMapEngine from './NodeMapEngine'

import { nodemapNodeSelected, nodemapNodeDeselected } from '../redux/actions'
import { DiagramModel } from "@projectstorm/react-diagrams"

interface Props {
  nodeSelected: any,
  nodeDeselected: any
};
interface States {
};

const mapStateToProps = (state) => ({})
const mapDispatchToProps = (dispatch) => ({
  nodeSelected: (payload: any) => dispatch(nodemapNodeSelected(payload)),
  nodeDeselected: (payload: any) => dispatch(nodemapNodeDeselected(payload)),
})

class NodeManager extends Component<Props, States> {
  // Link to singleton instance of nodemap graph engine
  private nodeMapEngine = NodeMapEngine.Instance;
  engine = this.nodeMapEngine.engine;
  
  constructor(props) {
    super(props);
    // Add listeners, noting the following useful resource:
    // https://github.com/projectstorm/react-diagrams/issues/164
    let model = this.engine.getModel(); 
    // Trigger a redux event from a listener on the first node
    model.getNodes().forEach(node =>
      node.registerListener({
	    selectionChanged: (e) => {
	      if (e.isSelected) {
		    this.props.nodeSelected(node.id);
          }
	    }
	  })
    );
  }

  render() {
    const engine = this.engine;
    return (
	  <div id="nodemanager" style={{width: '100%', height: '100%'}}>
      <BodyWidget engine={engine} />
      </div>
    )
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(NodeManager)
