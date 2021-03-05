# Compositional Image Generation with Lexicon Filters

## Training
```SHELL
  python main.py
```
See [main.py](main.py) for details for now...

## Models
- [VQVAE](src/vqvae.py)
- [LexGen](src/lex.py)
- [DAE](src/dae.py)
- [VAE](src/vae.py)

## Datasets \& Rendering

### Shapes
```SHELL
cd data/shapes
DISPLAY=:0 python render.py
```

### Set++
```SHELL
cd data/setpp
# npm install pug
node dataset.js
bash convert_png.sh # takes little long for now
```
### SCAN
1 - Download [images](https://drive.google.com/file/d/17khKEbQ0At0O7k4hM00i3hDxtNDVaXol/view\?usp\=sharing) from google drive   
2 - Extract images to `data/scan/images/`
```SHELL
cd data
python scan.py
```


## Visualizations
Currently all images dumped to [vis](vis/) folder.
