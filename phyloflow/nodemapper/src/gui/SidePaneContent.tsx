import React from 'react'
import { Component } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'
import { ReactSlidingPane } from 'react-sliding-pane'
import { displayCloseSettings } from "../redux/actions"

interface SidePaneProps {
  showpane: boolean;
  displayCloseSettings: any;
};
interface SidePaneState {};

const mapStateToProps = (state) => ({
    showpane: state.nodemap.display.show_settings_panel
})

const mapDispatchToProps = (dispatch) => ({
    displayCloseSettings: () => dispatch(displayCloseSettings())
})

class SidePaneContentComponent extends Component {
  constructor(props: {}) {
    super(props);
	// Business logic
  }

  componentDidMount() {
	// Business logic
  }

  render() {
    return (
		<>
		  <div>
		  List of parameters, etc.
		  </div>
		</>
    );
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(SidePaneContentComponent)
