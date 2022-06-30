const express = require('express');
const http = require('http');
// const spawn = require("child_process").spawn;
const { exec, execSync } = require("child_process");
const spawn = require('child_process').spawn;
const fs = require('fs');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require("path");
const serverless = require('serverless-http');
const multer = require('multer');

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

  const py = spawn("python", ["public/similarity_fast.py", `${dir}/test.txt`, `${dir}/database.xlsx`]);
  
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
    // fs.rmSync(dir, { recursive: true, force: true });
  });
 
});

app.get('/report/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const dir = './public/temp' + id.toString();

  res.download(path.resolve(`${dir}/report.pdf`)); 
})

app.post('/database/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const dir = './public/temp' + id.toString();
  fs.mkdirSync(dir);

  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, dir)
    },
    filename: function (req, file, cb) {
      cb(null, "database.xlsx" )
    }
  })

  const upload = multer({ storage: storage }).single('file');

  upload(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res.status(500).json(err)
    } else if (err) {
      return res.status(500).json(err)
    }
    return res.status(200).send(req.file)

  })

})

app.get("/", (req, res) => {
  res.send("This is the backend of the plagiarism checker")
})

const server = http.createServer(app);

server.listen(process.env.PORT || port, () => {
  console.log(`Server running at http://${hostname}:${port}`);
})
