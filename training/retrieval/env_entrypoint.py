import ray
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
   # Register the obsidian-retrieval environment
   # this needs to be done inside the entrypoint task
   register(
      id="obsidian-retrieval",  # <-- The name of the environment.
      entry_point="training.retrieval.env:RetrievalEnv",  # <-- The path to the environment class.
   )

   # make sure that the training loop is not run on the head node.
   exp = BasePPOExp(cfg)
   exp.run()

def main():
   # Parse command line arguments as Hydra overrides
   overrides = sys.argv[1:]
   
   # Create a base configuration
   base_cfg = OmegaConf.create({})
   
   # Apply the overrides to create final config
   cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))
   
   # validate the arguments
   validate_cfg(cfg)

   initialize_ray(cfg)
   ray.get(skyrl_entrypoint.remote(cfg))

if __name__ == "__main__":
   main()