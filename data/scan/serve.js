var express = require('express');
var serveIndex = require('serve-index');
var app = express();

//setting middleware
app.use(express.static("."));
app.use('/images', serveIndex(__dirname + '/images', { 'icons': true }));
var server = app.listen(8899);
