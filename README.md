# jax_IB
Immersed Boundary implementation in JAX-CFD


## Overview

This repository contains an implementation of an immersed boundary method in jax-cfd package. The immersed boundary method is a numerical technique used to simulate the interaction between fluid flow and immersed objects, such as particles, structures, or boundaries.

## Features

- Simulates transport of multiple rigid bodies 
- Combines Brownian dynamics integration with CFD for passive particle transport simulations

## Installation

To use a local version of the code, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/hashimmg/jax_ib.git
    ```

2. Navigate to the project directory:

    ```bash
    cd jax_ib
    ```

3. ```bash
   pip install -e .
   ```
   
### Example

The repository contains two examples

- Flapping of an ellipse airfoi
- Mixing in Journal bearing



