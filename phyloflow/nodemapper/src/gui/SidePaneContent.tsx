import React from 'react'
import { useAppSelector } from '../redux/store/hooks'
import { useSelector } from 'react-redux'

import "./SidePaneContent.css"

function SidePaneContentComponent() {
  const codesnippet = useAppSelector(state => state.display.codesnippet);
  return (
    <div>
    <p>List of parameters, etc.</p>
    <br/>
    <p>Code snippet<br/>
    <textarea id="codesnippet" {...{rows: 10}} style={{width: "100%"}} value={codesnippet} onChange={()=>{}} />
    </p>
    <button className="btn" style={{padding: "10px", float: "right"}}>SAVE AND RELOAD</button>
    </div>
  );
}

export default SidePaneContentComponent
