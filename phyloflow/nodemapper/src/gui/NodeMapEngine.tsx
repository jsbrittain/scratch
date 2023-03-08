import NodeScene from './NodeScene'

export default class NodeMapEngine {
  // Set up a singleton instance of a class
  private static _Instance: NodeMapEngine;
  nodeScene = null;
  engine = null;
  lock = false;
  
  constructor() {
    this.nodeScene = new NodeScene();
    this.engine = this.nodeScene.engine;
  }
  
  public static get Instance(): NodeMapEngine {
    return NodeMapEngine._Instance || (this._Instance = new this());
  } 

  public NodesSelectNone() {
    this.engine.getModel().getNodes().forEach(item => {
      item.setSelected(false);
    });
  }
  
  public LoadScene() {
    var input = document.createElement('input');
    input.type = 'file';
    input.onchange = e => {
      console.log(e);
      var file = (e.target as HTMLInputElement).files[0];
      
      var reader = new FileReader();
      reader.readAsText(file,'UTF-8');
      reader.onload = readerEvent => {
        var content = readerEvent.target.result;
		this.nodeScene.loadModel(content);
      }
    }
    input.click();
  }

  public SaveScene() {
    var str = this.nodeScene.serializeModel();
    this.download('model.json', str);
  }
  
  public download(filename, text) {	
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }

  public RunScene() {
    alert("Running the scene isn't supported just yet!");
  }

  public ToggleLock() {
    this.lock = !this.lock;
    if (this.lock)
	  document.getElementById("btnLock").innerHTML = "LOCK: ON";
    else
	  document.getElementById("btnLock").innerHTML = "LOCK: OFF";
  }
}
