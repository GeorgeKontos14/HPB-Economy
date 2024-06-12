import time
from Prepare import *
from Steps import *
from Draw import *
import PreComputed
import Store

pop_path = 'Data/pop_raw.csv'
yp_path = 'Data/yp_raw.csv'
burn_in = 5
total_draws = 40
save=True

def main():
    if save:
        Store.clear_files()
    start = time.time()
    # Prepare data
    precompute(pop_path, yp_path)
    end = time.time()
    print(f"Preparations: {end-start}")

    start = time.time()
    # Initialize Gibbs state
    initialize(
        PreComputed.regions, 
        PreComputed.no_kappas, 
        PreComputed.no_lambdas, 
        PreComputed.no_thetas
    )
    end = time.time()
    print(f"Initialization: {end-start}")

    start = time.time()
    sample(burn_in, total_draws)

    end = time.time()
    print(f"Gibbs Draws: {end-start}")

    if save:
        start = time.time()
        Store.write()
        end = time.time()
        print(f"Saving: {end-start}")

if __name__ == "__main__":
    main()