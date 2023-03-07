import React from 'react'
import { Component } from 'react'
import { render } from 'react-dom'
import "./Header.css"
import NodeMapEngine from './NodeMapEngine'

class Header extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <>
	  <link href="http://fonts.googleapis.com/css?family=Oswald" rel="stylesheet" type="text/css"/>
      <div style={{fontSize: 16, marginLeft: 0}}>PhyloFlow
	  <button className="btn" onClick={() => NodeMapEngine.Instance.LoadScene()}>LOAD</button>
	  <button className="btn" onClick={() => NodeMapEngine.Instance.SaveScene()}>SAVE</button>
	  <button className="btn" onClick={() => NodeMapEngine.Instance.RunScene()}>RUN</button>
      <button id="btnLock" className="btn" onClick={() => NodeMapEngine.Instance.ToggleLock()}>LOCK: OFF</button>
	  </div>
      </>
    )
  }
}

export default Header;
