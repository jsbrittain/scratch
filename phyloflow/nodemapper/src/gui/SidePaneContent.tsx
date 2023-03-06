import React from 'react'
import { Component } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'
import { ReactSlidingPane } from 'react-sliding-pane'
import { displayCloseSettings } from "../redux/actions"
import "./SidePaneContent.css"

interface SidePaneProps {
  showpane: boolean;
  displayCloseSettings: any;
};
interface SidePaneState {};

const mapStateToProps = (state) => ({
    showpane: state.display.show_settings_panel
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
		  <p>List of parameters, etc.</p>
		  <br/>
		  <p>Code snippet<br/>
		  <textarea id="Code snippet" name="sidePaneCodeSnippet" {...{rows: 10}} style={{width: "100%"}} ></textarea>
		  </p>
		  <button className="btn" style={{float: "right"}}>SAVE AND RELOAD</button>
		  </div>
		</>
    );
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(SidePaneContentComponent)
