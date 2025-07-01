import ray
import hydra
from omegaconf import DictConfig

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the multiply environment
    # this needs to be done inside the entrypoint task
    register(
        id="obsidian-retrieval",  # <-- The name of the environment.
        entry_point="training.retrieval.env:RetrievalEnv",  # <-- The path to the environment class.
    )

    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
