import time
from Prepare import *
from Steps import *
from Draw import *
import PreComputed

pop_path = 'Data/pop_raw.csv'
yp_path = 'Data/yp_raw.csv'
burn_in = 5
total_draws = 40

def main():
    
    # Prepare data
    precompute(pop_path, yp_path)

    # Initialize Gibbs state
    initialize(
        PreComputed.regions, 
        PreComputed.no_kappas, 
        PreComputed.no_lambdas, 
        PreComputed.no_thetas
    )
    start = time.time()
    for _ in range(10):
        draw()

    end = time.time()
    print(f"Time elapsed: {end-start}")

if __name__ == "__main__":
    main()