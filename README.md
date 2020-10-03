# Random Neighbors

Random Neighbors Clustering performs "random forest" style clustering on high-dimensional data. Provides sampling strategies
to bootstrap input rows and columns. Method iteratively fits clustering models to the 
samples and optimizes for the features that provide the best distinct cluster separation.


## Algorithm

Implementation is straight-forward:
- Bootstrap rows and columns based on inputted sampling strategy for a predefined set of iterations
- For each iteration on the sampled data, fit a clustering model and extract performance metrics
- Output the features that provide the most distinct clusters across all iterations

Like a random forest iteratively builds decision trees on bootstrapped data, random neighbors
iteratively builds clusters on bootstrapped data. Random Neighbors iterations can be used to extract the
single set of columns that provide largest cluster separation, or features can be averaged across iterations in a 
psuedo-feature importance metric. Dropout can be optionally applied to the history of iterations to avoid bias in 
the sampled fitting procedure. 

## Benefits

Main benefits include:
- Faster fitting on high-dimensional data through bootstrapping
- Iterations search for a low-dimensional representation of the data that produces distinct clusters
- Random sampling guards against correlation in input features, which can skew clustering results

## Case Study: Clustering Product Usage

Product logs contain a wealth of information on every action with a business application. Naturally,
business would like to segment users based on this activity in order to drive product strategy including pricing, new features,
churn reduction, and in-app upgrades. 

Product usage data can be extremely high-dimensional with a feature rich application. Size of 
input data grows with the total numbers of actions possible in the app and the total numbers
of users. The size and dimension of this data makes fitting clustering methods on the entire dataset challenging. 
Fitting on all rows is time-consuming, and a large amount of columns, possibly correlated, can build disoriented results. 

 


## Project Roadmap

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)