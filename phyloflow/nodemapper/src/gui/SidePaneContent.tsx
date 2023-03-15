import React from 'react'
import { useAppSelector } from '../redux/store/hooks'
import { useAppDispatch } from '../redux/store/hooks'
import { displayUpdateCodeSnippet } from '../redux/actions/display'
import { useSelector } from 'react-redux'

import "./SidePaneContent.css"

function SidePaneContentComponent() {
  const codesnippet = useAppSelector(state => state.display.codesnippet);
  const dispatch = useAppDispatch();

  const updateCodeSnippet = () => {
	// TODO: sort out payload
	const payload = ""
	dispatch(displayUpdateCodeSnippet(payload))
  }

  return (
    <div>
    <p>Description, environment, etc.</p>
    <br/>
    <p>Code snippet<br/>
    <textarea
      id="codesnippet" {...{rows: 10}}
      style={{width: "100%"}}
      value={codesnippet}
      onChange={()=>{}}
    />
    </p>
    <button
      className="btn"
      style={{padding: "10px", float: "right"}}
      onClick={updateCodeSnippet}
      disabled={true}
    >SAVE</button>
    </div>
  );
}

export default SidePaneContentComponent
