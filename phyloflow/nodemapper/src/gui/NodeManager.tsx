import React from 'react'
import { Component, StrictMode, useEffect } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'

import { BodyWidget } from './BodyWidget'
import './NodeManager.css'
import NodeScene from './NodeScene'

import { nodemapNodeSelected, nodemapNodeDeselected } from '../redux/actions'
import { DiagramModel } from "@projectstorm/react-diagrams"

interface Props {
  nodeSelected: any,
  nodeDeselected: any
};
interface States {
  nodeScene: NodeScene
};

const mapStateToProps = (state) => ({})
const mapDispatchToProps = (dispatch) => ({
  nodeSelected: () => dispatch(nodemapNodeSelected()),
  nodeDeselected: () => dispatch(nodemapNodeDeselected()),
})

class NodeManager extends Component<Props, States> {
  constructor(props) {
    super(props);
    this.state = { nodeScene: new NodeScene() }
    // Add listeners, noting the following useful resource:
    // https://github.com/projectstorm/react-diagrams/issues/164
    let model = this.state.nodeScene.engine.getModel();
    
    // Trigger a redux event from a listener on the first node
    model.getNodes()[0].registerListener({
	  selectionChanged: (e) => {
	    if (e.isSelected) {
		  this.props.nodeSelected();
	//var str = JSON.stringify(model.serialize());
	//model.deserializeModel(JSON.parse(str), this.state.nodeScene.engine);
	/*var model2 = new DiagramModel();
	model2.deserializeModel(JSON.parse(str), this.state.nodeScene.engine);
	this.state.nodeScene.engine.setModel(model2);*/
		} else {
          this.props.nodeDeselected();
		  //model.getNodes()[0].setPosition(0,0);
        }
	  },
	});
  }

  render() {
    const engine = this.state.nodeScene.engine;
    return (
	  <div id="nodemanager" style={{width: '100%', height: '100%'}}>
      <BodyWidget engine={engine} />
      </div>
    )
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(NodeManager)
