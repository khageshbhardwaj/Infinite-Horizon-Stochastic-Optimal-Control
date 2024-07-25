# UCSD ECE276B PR3

# Project Overview
This project encompasses implementations for two computational experiments: Certainty Equivalent Control (CEC) and General Policy Iteration (GPI). Below are detailed instructions for setting up and executing each component.

## Project Structure
The codebase is split into two main segments:
- **CEC Implementation**: Includes `main_cec.py` which leverages the `cec.py` module, alongside the utility functions provided in `utils.py`.
- **GPI Implementation**: Managed by the `main_gpi.py` script, executing the GPI algorithm.

## Installation
Before executing the scripts, you must install the necessary dependencies. This can be accomplished via the following command:
pip install -r requirements.txt

## Starter code
### 1. main_cec.py
This file contains main code to run the CEC algorithm.

### 2. utils.py
This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.

### 3. cec.py
This file provides code for the CEC algorithm (Part 1 of the project).
Change obstacle_avoidance = False to obstacle_avoidance = True to consider obstacle in the environment. 
Also while running with obstacle_avoidance = True program might terminate with infeasible solution warning. In that case, rerun the program again (you are guaranteed to find a solution).

### 4. gpi.py
This file provides code for the GPI algorithm (Part 2 of the project).

