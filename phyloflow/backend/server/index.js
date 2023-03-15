const express = require("express");

const PORT = process.env.PORT || 3001;
const app = express();

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

app.listen(PORT, () => {
	  console.log(`Server listening on ${PORT}`);
});
