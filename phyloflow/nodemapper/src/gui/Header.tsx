import React from 'react'
import { Component } from 'react'
import { render } from 'react-dom'
import "./Header.css"

class Header extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <>
	  <link href="http://fonts.googleapis.com/css?family=Oswald" rel="stylesheet" type="text/css"/>
      <div style={{fontSize: 16, marginLeft: 0}}>PhyloFlow
	  <button className="btn">RUN</button>
	  </div>
      </>
    )
  }
}

export default Header;
