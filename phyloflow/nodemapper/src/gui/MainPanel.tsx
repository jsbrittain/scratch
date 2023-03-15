import React from 'react'
import NodeManager from './NodeManager'

function MainPanel() {
  const [data,setData] = React.useState(null);
  const requestOptions = {
    method: 'GET',
    headers: {'Content-Type': undefined},
  }
  React.useEffect(() => {
    fetch('http://127.0.0.1:3001/api', requestOptions)
      .then((res) => res.json())
      .then((data) => setData(data.message));
  }, [])  // run once

  return (
    <>
    <p>{!data ? "Loading..." : data}</p>
    </>
  )
}

export default MainPanel;


//<NodeManager />
