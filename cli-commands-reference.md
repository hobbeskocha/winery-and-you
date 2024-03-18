# Jupytext for py to ipynb, and vice versa 
jupytext --to .py --output {output-filepath} {notebook-input-filepath}
jupytext --to .ipynb --output {output-filepath} {py-input-filepath}

# nbconvert for ipynb to html
jupyter nbconvert {input-filepath} --ouput-dir="{filepath}" --to html
