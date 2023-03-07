// Imports for panel-manager
import React from 'react'
import { Component, StrictMode, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { action } from '@storybook/addon-actions'
import createEngine, { DiagramModel, DefaultNodeModel, DefaultLinkModel, DiagramEngine } from '@projectstorm/react-diagrams'
import { BodyWidget } from './BodyWidget'

class NodeScene {
  engine: DiagramEngine;
  
  constructor() {
    this.InitializeScene();
  }

  addNode(name, color, pos) {
    var node = new DefaultNodeModel(name, color);
    node.setPosition(pos[0], pos[1]);
    this.engine.getModel().addNode(node);
    return node;
  }

  addLink(port_from, port_to) {
    const link = new DefaultLinkModel();
    link.setSourcePort(port_from);
    link.setTargetPort(port_to);
    this.engine.getModel().addLink(link);
  }
  
  InitializeScene() {
    // Initialise Node drawing engine and specify starting layout
    this.engine = createEngine();
    const model = new DiagramModel();
    this.engine.setModel(model);
    
    var node1 = this.addNode('Input', 'rgb(192,255,0)', [400, 118]);
    node1.addOutPort('out-1');
    node1.addOutPort('out-2');
    node1.addOutPort('out-3');
    node1.addOutPort('out-4');

    var node2 = this.addNode('Process 1', 'rgb(0,192,255)', [525, 100]);
    node2.addInPort('in-1');
    node2.addInPort('in-2');
    node2.addOutPort('out-1');
    node2.addOutPort('out-2');

    var node3 = this.addNode('Process 2', 'rgb(0,192,255)', [700, 80]);
    node3.addInPort('in-1');
    node3.addInPort('in-2');
    
	var node4 = this.addNode('Logging', 'rgb(192,0,255)', [700, 150]);
    node4.addInPort('in-1');
    node4.addInPort('in-2');
    node4.addInPort('in-3');
    node4.addInPort('in-4');

    this.addLink(node1.getPort('out-1'), node2.getPort('in-1'));
    this.addLink(node2.getPort('out-1'), node3.getPort('in-1'));
    this.addLink(node2.getPort('out-2'), node4.getPort('in-1'));
    this.addLink(node1.getPort('out-4'), node4.getPort('in-2'));
  }
}

export default NodeScene;
