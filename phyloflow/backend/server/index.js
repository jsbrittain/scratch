const express = require("express");
const bodyParser = require("body-parser");

const PORT = process.env.PORT || 3001;
const app = express();

var jsonParser = bodyParser.json()

// Configure access control
app.use(function (req, res, next) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Credentials', true);
  next();
});

app.get("/api", (req, res) => {
  res.json({ message: "Hello from server!"});
});

app.post("/tokenize", jsonParser, (req, res) => {
  console.log("POST /tokenize")
  console.log(req.body)
  res.writeHead(200,{'Content-Type': 'application/json;charset=UTF-8'})
  res.end(JSON.stringify({
    response: "hello from server"
  }))
});

app.listen(PORT, () => {
  console.log(`Server listening on ${PORT}`);
});
