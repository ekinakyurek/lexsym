const pug = require('pug');
const fs = require('fs');
const Card = pug.compileFile('card-svg.pug');
const shapes = ['Diamond', 'Squiggle', 'Pill'];
const fills  = ['Hatched', 'Solid', 'Open'];
// const colors = ['#2c3e50', '#f1c40f', '#2ecc71'];
const colors = ['red', 'blue', 'green'];
const cartesian =
  (...a) => a.reduce((a, b) => a.flatMap(d => b.map(e => [d, e].flat())));

var types = cartesian(shapes,colors,fills).map(x => { return {shape: x[0], color: x[1], fill: x[2]}});

const str =
  (ttype) => "s_" + ttype.shape[0] + "_c_" + ttype.color[0] + "_f_" + ttype.fill[0];

const lstr =
  (ttype) =>  ttype.fill + " " + ttype.color + " " + ttype.shape;

const hidden = "s_S_c_r_f_S"
const data   = []
const splits = {"train":[], "test":[]}
all = types.map(x =>[x]).concat(cartesian(types,types)).concat(cartesian(types,types,types))
var id=0;

for (const type of all) {
   var card = Card({objs:type});
   var name = type.map(x=>str(x)).join("_");
   var text = type.map(x=>lstr(x)).join(" , ").toLowerCase();
   fs.writeFileSync("images/"+name+".svg",card);
   data.push({"image": "images/"+name+".png", "text": text});
   if (name.includes(hidden)){
      splits["test"].push(id);
   }else{
      splits["train"].push(id);
   }
   id = id + 1;
}

var datastring = JSON.stringify(data);
console.log(datastring);
fs.writeFileSync("data.json", datastring);

var splitstring = JSON.stringify(splits);
fs.writeFileSync("splits.json", splitstring);