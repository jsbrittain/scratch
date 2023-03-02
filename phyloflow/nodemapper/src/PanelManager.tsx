// Imports for panel-manager
import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { action } from '@storybook/addon-actions';
import createEngine, { DiagramModel, DefaultNodeModel, DefaultLinkModel, DiagramEngine } from '@projectstorm/react-diagrams';
import { BodyWidget } from './BodyWidget';

// Imports for node-manager
import { Component, StrictMode, useRef, useState } from 'react';
import { render } from 'react-dom';
import { ReactSlidingPane } from 'react-sliding-pane';
import './PanelManager.css';

// Initialise Node drawing engine and specify layout (here for now)
const engine = createEngine();
var model = new DiagramModel();
var node1 = new DefaultNodeModel('Input', 'rgb(192,255,0)');
node1.setPosition(200, 100);
node1.addOutPort('out-4', false);
node1.addOutPort('out-3', false);
node1.addOutPort('out-2', false);
node1.addOutPort('out-1', false);

var node2 = new DefaultNodeModel('Process', 'rgb(0,192,255)');
node2.setPosition(325, 100);
node2.addInPort('in-2', false);
node2.addInPort('in-1', false);
node2.addOutPort('out-2', false);
node2.addOutPort('out-1', false);

var node3 = new DefaultNodeModel('Output', 'rgb(192,0,255)');
node3.setPosition(500, 140);
node3.addInPort('in-2', false);
node3.addInPort('in-1', false);

const link1 = new DefaultLinkModel();
link1.setSourcePort(node1.getPort('out-1'));
link1.setTargetPort(node2.getPort('in-1'));

const link2 = new DefaultLinkModel();
link2.setSourcePort(node2.getPort('out-2'));
link2.setTargetPort(node3.getPort('in-1'));

const link3 = new DefaultLinkModel();
link3.setSourcePort(node1.getPort('out-4'));
link3.setTargetPort(node3.getPort('in-2'));

let models = model.addAll(node1, node2, node3, link1, link2, link3);

// Add a selection listener to each model element
models.forEach((item) => {
	item.registerListener({
		eventDidFire: action('element eventDidFire')
	});
});

model.registerListener({
	eventDidFire: action('model eventDidFire')
});

engine.setModel(model);


// Layout for main window, including sliding-pane support
export default function App() {
  const rootRef = useRef<HTMLDivElement | null>();
  const [paneOpened, setOpenedPane] = useState<'right'|'left'|'bottom'|null>(null);

  return (
    <StrictMode>
      <div ref={rootRef}>
		<div id="header-panels">
			<button onClick={() => setOpenedPane('left')}>
			  Open settings pane
			</button>
		</div>
		<div id="main-panels">
			<div id="nodemanager" style={{width: '100%', height: '800px'}}><BodyWidget engine={engine} /></div>

			{/* Left-side pane (notionally for parameter settings) */}
			<ReactSlidingPane
			  className="some-custom-class"
			  overlayClassName="some-custom-overlay-class"
			  from="left"
			  width="25%"
			  isOpen={paneOpened === 'left'}
			  title="Parameters / settings"
			  subtitle="Node parameters here"
			  onRequestClose={() => {
				// triggered on "<" on left top click or on outside click
				setOpenedPane(null);
			  }}
			>
			<Content />
			</ReactSlidingPane>

		</div>
      </div>
    </StrictMode>
  );
}

class Content extends Component {
  constructor(props: {}) {
    super(props);
    // eslint-disable-next-line
    console.log("contructor");
  }

  componentDidMount() {
    // eslint-disable-next-line
    console.log("mount");
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
