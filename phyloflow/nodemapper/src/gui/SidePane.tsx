import React from 'react'
import { ReactSlidingPane } from 'react-sliding-pane'
import { displayCloseSettings } from '../redux/actions'
import SidePaneContent from './SidePaneContent'
import { useAppSelector, useAppDispatch } from '../redux/store/hooks'
import './SidePane.css'

function SidePane() {
  const showpane = useAppSelector(state => state.display.show_settings_panel);
  const title = useAppSelector(state => state.display.settings_title);
  const dispatch = useAppDispatch();
  return (
    <ReactSlidingPane
      className="some-custom-class"
      overlayClassName="some-custom-overlay-class"
      from="left"
      width="33%"
      isOpen={showpane}
      title={title}
      subtitle="Placeholder subtitle"
      onRequestClose={() => {
        // triggered on "<" on left top click or on click outside of pane
        dispatch(displayCloseSettings());
      }}
    >
    <SidePaneContent />
    </ReactSlidingPane>
  )
};

export default SidePane
