// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

// [START gae_node_request_example]
const express = require('express');

require('dotenv').config();
// console.log('scipy-optimize');

// opt.localMinimize(function(x) {
//   return Math.pow(x-4, 4) - Math.pow(x, 3) + 10 * x - 1;
// }, {
//   bounds: [-10, 10]
// }, function(results) {
//   console.log(results);
// });

const app = express();
var opt = require('./scipy').opt;
// var  opt = require('scipy-optimize');

app.get('/', (req, res) => {
  res.status(200).send('Hello, world!').end();
});

// Start the server
const PORT = parseInt(process.env.PORT) || 8080;
// app.listen(PORT, () => {
//   console.log(`App listening on port ${PORT}`);
//   console.log('Press Ctrl+C to quit.');
// });
// [END gae_node_request_example]


////////////////////////////////////////////

app.use(function (req, res, next) {

  // Website you wish to allow to connect
  res.setHeader('Access-Control-Allow-Origin', 'http://localhost:3000');

  // Request methods you wish to allow
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');

  // Request headers you wish to allow
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,content-type');

  // Set to true if you need the website to include cookies in the requests sent
  // to the API (e.g. in case you use sessions)
  res.setHeader('Access-Control-Allow-Credentials', true);

  // Pass to next layer of middleware
  next();
});

// app.get("/", (req, res) => res.send("Hello World!!!"));
app.get("/stock/:code/:model", (req, res) => {
  opt.minimize(function() {}, {
    code: req.params.code,
    model: req.params.model,
  }, function(results) {
    console.log('results', results);
    res.json([results]);
  });
})

app.get("/products", (req, res) => {
  opt.localMinimize(function(x) {
    // return x+10;/
    return Math.pow(x-4, 4) - Math.pow(x, 3) + 10 * x - 1;
  }, {
    bounds: [-10, 10]
  }, function(results) {
    console.log('results', results);
    res.json([results]);
  });

})
app.listen(PORT, () => console.log(`Example app listening on port ${PORT}!`));



module.exports = app;
