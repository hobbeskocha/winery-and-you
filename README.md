# Winery and You!

This is a collaborative personal portfolio project between [Hyewon Jeong](https://www.linkedin.com/in/jeonghyewon/) and [Ayush Shrestha](https://www.linkedin.com/in/ayush-yoshi-shrestha/). The dataset used is on sales order data for a national winery, accessible [here](data/). The dataset was provided to us by [Dr. Oliver J. Rutz](https://foster.uw.edu/faculty-research/directory/oliver-rutz/) of the University of Washington, Seattle.

## Instructions
For easy version control purposes, the source code is stored as converted Python scripts, but HTML files of the source Python notebooks can be found [here](src/html/).

The analysis performed consists of initial data cleaning processing of the dataset using Pandas. This consisted of handling missing values, removing erroneous data (such as invalid US State abbreviations), and creating a new dataframe that contained aggregated data to capture each unique customer's information. Some preliminary EDA was performed in Seaborn and Matplotlib, to look at the shape and distribution of the data. This was expanded upon in a packaged Tableau workbook, yielding novel understanding of customer and order sales figures across the United States as well as over the lifetime of the dataset.

The predictive analysis is performed using Statsmodels and Scikit-learn to build predictive models for customers' subscription preferences of email, newsletter, and winemaker calls. The models used were Logistic Regression and Random Forest, namely becuase they offered insight into the weight/importance of each independent variable via coefficients and feature importances. This includes generating models metrics like accuracy, precision, and AUC to assess the predictive strength of the models. Confusion matrices were also generated to get a better sense of which cases that particular models were performing well on.
We then went one step further and calculated the lift of the subscriptions. This allowed generating marginal response and positive response curves to understand how customers would respond to solicitation for subscription.
