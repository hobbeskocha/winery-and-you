# Jupytext for py to ipynb, and vice versa 
jupytext --to .py --output {output-filepath} {notebook-input-filepath}
    jupytext --to .py --output src/1_data_cleaning.py src/1_data_cleaning.ipynb
    jupytext --to .py --output src/2_descriptive_analysis.py src/2_descriptive_analysis.ipynb
    jupytext --to .py --output src/3_predictive_analysis.py src/3_predictive_analysis.ipynb

jupytext --to .ipynb --output {output-filepath} {py-input-filepath}

# nbconvert for ipynb to html
jupyter nbconvert {input-filepath} --ouput-dir="{filepath}" --to html
