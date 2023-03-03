// Imports for panel-manager
import React from 'react'
import { Component, StrictMode, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { action } from '@storybook/addon-actions'
import createEngine, { DiagramModel, DefaultNodeModel, DefaultLinkModel, DiagramEngine } from '@projectstorm/react-diagrams'
import { BodyWidget } from './BodyWidget'

export default function InitializeScene() {
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
            eventDidFire: () => { console.log('element eventDidFire'); }
        });
    });

    model.registerListener({
        eventDidFire: () => console.log('model eventDidFire')
    });

    engine.setModel(model);
    return engine
}
