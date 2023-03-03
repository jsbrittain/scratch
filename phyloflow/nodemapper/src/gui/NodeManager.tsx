import React from 'react'
import { Component, StrictMode } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'

import { BodyWidget } from './BodyWidget'
import "./NodeManager.css"
import { nodemapInitializeEngine } from '../redux/actions'

interface Props {
  engine: any,
  initializeEngine: any
};
interface States {};

const mapStateToProps = (state) => ({
  engine: state.nodemap.engine,
})

const mapDispatchToProps = (dispatch) => ({
  initializeEngine: () => dispatch(nodemapInitializeEngine)
})

class NodeManager extends Component<Props, States> {
  constructor(props) {
    super(props);
  }

  render() {
    const { engine } = this.props;
    return (
	  <div id="nodemanager" style={{width: '100%', height: '100%'}}>
      <BodyWidget engine={engine} />
      </div>
    )
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(NodeManager)
