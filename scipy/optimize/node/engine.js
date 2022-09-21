var clean = require('./clean');

module.exports = Engine = {};

Engine.runPython = function(operation, a, b, cb, x, y) {
  if (operation === 'minimize') {
    console.log(1, 'minimize');
    console.log(2, b);
    var spawn = require('child_process').spawn;
    var env = Object.create(process.env);
    env.PYTHONUNBUFFERED = '1';
    
    var python = spawn(
    process.env.PYTHON_PATH,
    ['-u', __dirname + '\\..\\py\\var_garch.py', b.code, b.model, process.env.CSV_PATH, process.env.LIST_PATH], 
    {'env': env});
    var output = '';
    python.stdout.on('data', function (data){
      console.log(new Date(), data.toString());
      output += data.toString();
    });
    python.on('close', function (){
      try {
        cb(JSON.parse(output));
      } catch (e) {
        cb(output);
      }
    });
  } else {
    if (operation === 'local' || operation === 'global') {
      var cleanup = clean.cleanMin(operation, a, b, cb);
      a   = cleanup.func;
      b = JSON.stringify(cleanup.options);
      cb  = cleanup.callback;
    } else if (operation === 'nnls') {
      cb = clean.cleanCB(cb);
      a = JSON.stringify(a);
      b = JSON.stringify(b);
    } else if (operation === 'fit') {
      var cleanup = clean.cleanFit(a, b, cb, x, y);
      a = cleanup.func;
      b = JSON.stringify(cleanup.options);
      cb = cleanup.callback;
    } else if (operation === 'root') {
      var cleanup = clean.cleanRoot(a, b, cb, x, y);
      a = cleanup.func;
      b = JSON.stringify(cleanup.options);
      cb = cleanup.callback;
    } else if (operation === 'vectorRoot') {
      var cleanup = clean.cleanVector(a, b, cb, x);
      a = cleanup.func;
      b = JSON.stringify(cleanup.options);
      cb = cleanup.callback;
    } else if (operation === 'derivative') {
      var cleanup = clean.cleanDerivative(a, b, cb, x);
      a = cleanup.func;
      b = JSON.stringify(cleanup.options);
      cb = cleanup.callback;
    }

    // don't need to worry about race conditions with async process below
    // since each is wrapped in their own "runPython" closure
    var spawn = require('child_process').spawn;
    var env = Object.create(process.env);
    env.PYTHONUNBUFFERED = '1';
    
    var python = spawn(
    env.PYTHON_PATH,
    ['-u', __dirname + '\\..\\py\\exec.py', operation, a, b],
    {'env': env});
    var output = '';
    python.stdout.on('data', function (data){
      output += data.toString();
    });
    python.on('close', function (){
      try {
        cb(JSON.parse(output));
      } catch (e) {
        cb(output);
      }
    });
  }
}
