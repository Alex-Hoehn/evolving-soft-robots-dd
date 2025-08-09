# evolving-soft-robots-dd
This code repository contains the novel contributions introduced in the Master thesis "Evolving Soft Robots Via Discrete Diffusion models".

The code itself has largely been adapted from both the original Evogym repository (https://github.com/EvolutionGym/evogym), 
as well as the Score Entropy Discrete Diffusion repository (https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

For the background and general approach of this implementation, please refer to the thesis.

# Setup
These are the steps to set up the environment and run the code:
1. Clone the repository and setup according to the instructions in the original Evogym repository (https://github.com/EvolutionGym/evogym).
2. Replace the `PhysicsEngine.cpp` file in the `evogym/simulator` directory with the one provided in this repository.
3. Run the following command to build the new physics engine:
   ```bash
   python setup.py build_ext --inplace
   ```
4. Insert the files given in the `examples` directory in this repository into the `examples` directory of the original Evogym repository.
5. Ready to run, example test run:
   ```bash
   cd examples
   python run_sedd.py --exp-name "walker_sedd_test" --env-name "Walker-v0" --num-robots 25 --cycles 4 --num-generated 1000 --fine-tune-epochs 3 --total-timesteps 256000 --learning-rate 2.5e-4 --clip-range 0.1 --vf-coef 0.5 --ent-coef 0.01 --n-steps 128 --n-envs 1 --eval-interval 12800 --num-cores 2
   ```
