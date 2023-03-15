import React from 'react'
import { Component } from 'react'
import { render } from 'react-dom'
import NodeManager from './NodeManager'

class MainPanel extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <>
      <NodeManager />
      </>
    )
  }
}

export default MainPanel;
