# Optimization
Scripts for TTF optimizations

## Files
- connect.py - connects to Gainy research database, runs queries, gets EOD Index data. Needs environment variables for access
- my_utils.py - date utilities
- portfolio.py - portfolio functions
- optimizer - optimizer class. Performs two types of optimization: Sharpe and Risk budget - we use risk budget so far
- ttf_optimizer - main script. Takes parameters and runs an optimization that is written in a csv file

## Running the main script

python ttf_optimizer.py -i <id> -d <date> -i <output file> 

Run 
python ttf_optimizer.py -h       for help

