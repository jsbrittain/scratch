import React from 'react'
import { useState } from 'react'
import { useEffect } from 'react'
import { useAppSelector } from '../redux/store/hooks'
import { useAppDispatch } from '../redux/store/hooks'
import { displayToggleGraphMoveable } from '../redux/actions'
import { nodemapSubmitQuery } from '../redux/actions'
import NodeMapEngine from './NodeMapEngine'
import "./Header.css"

function Header() {
  const [textEditGraph, setTextEditGraph] = useState("EDIT GRAPH: OFF");
  const graph_is_moveable = useAppSelector(state => state.display.graph_is_moveable);
  const dispatch = useAppDispatch();

  // === Load Scene ===========================================================

  const btnLoadScene = () => {
    NodeMapEngine.Instance.LoadScene()
  }
  
  // === Save Scene ===========================================================
  
  const btnSaveScene = () => {
    NodeMapEngine.Instance.SaveScene()
  }
  
  // === Import Snakefile =====================================================

  const btnImportSnakefile = () => {
    dispatch(nodemapSubmitQuery('TEST QUERY'))
    NodeMapEngine.Instance.ImportSnakefile();
  }
  
  // === Build Snakefile ======================================================

  const btnBuildSnakefile = () => {
    NodeMapEngine.Instance.BuildSnakefile();
  }

  // === Toggle graph moveability =============================================

  // Dispatch action to toggle graph moveability state...
  const btnToggleLock = () => {
    dispatch(displayToggleGraphMoveable())
  }
  // ... then react to state change by updating button text
  useEffect(() => {
    if (graph_is_moveable)
      setTextEditGraph("EDIT GRAPH: ON")
    else
      setTextEditGraph("EDIT GRAPH: OFF")
  }, [graph_is_moveable])

  // ==========================================================================
  
  return (
    <>
    <link href="http://fonts.googleapis.com/css?family=Oswald" rel="stylesheet" type="text/css"/>
    <div style={{fontSize: 16, marginLeft: 0}}>PhyloFlow
      <button className="btn" onClick={btnLoadScene}>LOAD</button>
      <button className="btn" onClick={btnSaveScene}>SAVE</button>
      <button className="btn" onClick={btnImportSnakefile}>IMPORT SNAKEFILE</button>
      <button className="btn" >BUILD SNAKEFILE</button>
      <button className="btn" onClick={btnToggleLock}>{textEditGraph}</button>
    </div>
    </>
  )
}

export default Header;
