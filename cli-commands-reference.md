# Jupytext for py to ipynb, and vice versa

jupytext --to .py --output {output-filepath} {notebook-input-filepath}
    jupytext --to .py --output src/1_data_cleaning.py src/1_data_cleaning.ipynb
    jupytext --to .py --output src/2_descriptive_analysis.py src/2_descriptive_analysis.ipynb
    jupytext --to .py --output src/3_predictive_analysis.py src/3_predictive_analysis.ipynb

jupytext --to .ipynb --output {output-filepath} {py-input-filepath}
    jupytext --to .ipynb --output src/1_data_cleaning.ipynb src/1_data_cleaning.py
    jupytext --to .ipynb --output src/2_descriptive_analysis.ipynb src/2_descriptive_analysis.py
    jupytext --to .ipynb --output src/3_predictive_analysis.ipynb src/3_predictive_analysis.py

# nbconvert for ipynb to html

jupyter nbconvert {input-filepath} --output-dir="{filepath}" --to html
    jupyter nbconvert src/1_data_cleaning.ipynb --output-dir="src/html" --to html
    jupyter nbconvert src/2_descriptive_analysis.ipynb --output-dir="src/html" --to html
    jupyter nbconvert src/3_predictive_analysis.ipynb --output-dir="src/html" --to html
