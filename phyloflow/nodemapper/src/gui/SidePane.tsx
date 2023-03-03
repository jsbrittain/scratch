import React from 'react'
import { Component } from 'react'
import { connect } from 'react-redux'
import { render } from 'react-dom'
import { ReactSlidingPane } from 'react-sliding-pane'
import { displayCloseSettings } from '../redux/actions'
import SidePaneContent from './SidePaneContent'
import './SidePane.css'

interface Props {
  showpane: boolean;
  displayCloseSettings: any;
};
interface State {};

const mapStateToProps = (state) => ({
    showpane: state.display.show_settings_panel
})

const mapDispatchToProps = (dispatch) => ({
    displayCloseSettings: () => dispatch(displayCloseSettings())
})

class SidePane extends Component<Props, State> {
  constructor(props) {
    super(props);
  }

  render() {
    const { showpane } = this.props;
    return (
      <>
      <ReactSlidingPane
        className="some-custom-class"
        overlayClassName="some-custom-overlay-class"
        from="left"
        width="25%"
        isOpen={showpane}
        title="Parameters / settings"
        subtitle="Node parameters here"
        onRequestClose={() => {
          // triggered on "<" on left top click or on outside click
          this.props.displayCloseSettings()
        }}
      >
      <SidePaneContent />
      </ReactSlidingPane>
      </>
    )
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(SidePane)
