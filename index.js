const express = require('express');
const http = require('http');
// const spawn = require("child_process").spawn;
const { exec, execSync } = require("child_process");
const spawn = require('child_process').spawn;
const fs = require('fs');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require("path");

const hostname = 'localhost';
const port = 4000;

const app = express();

app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.post('/text/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const dir = './public/temp' + id.toString();
  const text = req.body.text;

  fs.mkdirSync(dir);
  fs.writeFile(`${dir}/test.txt`, text, (err) => {
    if (err) {
      console.log(err);
    }
  });
  return res.end();
})

app.get('/process/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const dir = './public/temp' + id.toString();

  // exec(`python public/similarity.py ${dir}/test.txt public/test.xlsx`, (error, stdout, stderr) => {
  //   if (error) {
  //       console.log(`error: ${error.message}`);
  //       return;
  //   }
  //   if (stderr) {
  //       console.log(`stderr: ${stderr}`);
  //       return;
  //   }
  // });  

  const py = spawn("python", ["public/similarity.py", `${dir}/test.txt`, "public/test.xlsx"]);
  
  py.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  py.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });
  
  py.on('exit', (code) => {
    fs.renameSync("report.pdf", `${dir}/report.pdf`, (err) => {
      if (err) {
        console.log(err);
      }
    });    
    res.download(path.resolve(`${dir}/report.pdf`)); 
    fs.rmSync(dir, { recursive: true, force: true });
  });
 
});

app.get('/report/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const dir = './public/temp' + id.toString();

  res.download(path.resolve(`${dir}/report.pdf`)); 
})

const server = http.createServer(app);

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}`)
})
