## Table of contents
   1. [Installation](#installation)
   1. [Example](#example)
   1. [Documentation](#documentation)
   1. [Details](#details)
   1. [Data](#data)
   1. [Baselines](#baselines)
   1. [Acknowledgments](#acknowledgments)
   1. [References](#references)

## Installation

1. Clone a stable version from the github repository and install habitat-lab using the commands below. Note that python>=3.7 is required for working with habitat-lab. All the development and testing was done using python3.7. Please use 3.7 to avoid possible issues.

    ```bash
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -e .
    ```

    The command above will install only core of Habitat Lab. To include habitat_baselines along with all additional requirements, use the command below instead:

    ```bash
    git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
    cd habitat-lab
    pip install -r requirements.txt
    python setup.py develop --all # install habitat and habitat_baselines
    ```

2. Install `habitat-sim`:

      For a machine with an attached display,

      ```bash
      conda install habitat-sim withbullet -c conda-forge -c aihabitat
      ```

      For a machine with multiple GPUs or without an attached display (i.e. a cluster),

      ```bash
       conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
      ```

      See habitat-sim's [installation instructions](https://github.com/facebookresearch/habitat-sim#installation) for more details.
      MacOS does *not* work with `headless` so exclude that argument if on MacOS.


3. Run the example script `python examples/example.py ` which in the end should print out number of steps agent took inside an environment (eg: `Episode finished after 18 steps.`).


## Example
<!--- Please, update `examples/example.py` if you update example. -->

🆕Example code-snippet which uses [`configs/tasks/rearrange/pick.yaml`](configs/tasks/rearrange/pick.yaml) for configuration of task and agent.

```python
import habitat

# Load embodied AI task (RearrangePick) and a pre-specified virtual robot
env = habitat.Env(
    config=habitat.get_config("configs/tasks/rearrange/pick.yaml")
)

observations = env.reset()

# Step through environment with random actions
while not env.episode_over:
    observations = env.step(env.action_space.sample())

```

See [`examples/register_new_sensors_and_measures.py`](examples/register_new_sensors_and_measures.py) for an example of how to extend habitat-lab from _outside_ the source code.

## Documentation

Habitat Lab documentation is available [here](https://aihabitat.org/docs/habitat-lab/index.html).

For example, see [this page](https://aihabitat.org/docs/habitat-lab/quickstart.html) for a quickstart example.


## Docker Setup
We also provide a docker setup for habitat. This works on machines with an NVIDIA GPU and requires users to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). The following [Dockerfile](Dockerfile) was used to build the habitat docker. To setup the habitat stack using docker follow the below steps:

1. Pull the habitat docker image: `docker pull fairembodied/habitat:latest`

1. Start an interactive bash session inside the habitat docker: `docker run --runtime=nvidia -it fairhabitat/habitat:v1`

1. Activate the habitat conda environment: `source activate habitat`

1. Benchmark a forward only agent on the test scenes data: `cd habitat-api; python examples/benchmark.py`. This should print out an output like:
```bash
2019-02-25 02:39:48,680 initializing sim Sim-v0
2019-02-25 02:39:49,655 initializing task Nav-v0
spl: 0.000
```

## Details
An important objective of Habitat Lab is to make it easy for users to set up a variety of embodied agent tasks in 3D environments. The process of setting up a task involves using environment information provided by the simulator, connecting the information with a dataset (e.g. PointGoal targets, or question and answer pairs for Embodied QA) and providing observations which can be used by the agents. Keeping this primary objective in mind the core API defines the following key concepts as abstractions that can be extended:

* `Env`: the fundamental environment concept for Habitat. All the information needed for working on embodied tasks with a simulator is abstracted inside an Env. This class acts as a base for other derived environment classes. Env consists of three major components: a Simulator, a Dataset (containing Episodes), and a Task, and it serves to connects all these three components together.

* `Dataset`: contains a list of task-specific episodes from a particular data split and additional dataset-wide information. Handles loading and saving of a dataset to disk, getting a list of scenes, and getting a list of episodes for a particular scene.

* `Episode`: a class for episode specification that includes the initial position and orientation of an Agent, a scene id, a goal position and optionally shortest paths to the goal. An episode is a description of one task instance for the agent.

<p align="center">
  <img src='res/img/habitat_lab_structure.png' alt="teaser results" width="100%"/>
  <p align="center"><i>Architecture of Habitat Lab</i></p>
</p>

* `Task`: this class builds on top of the simulator and dataset. The criteria of episode termination and measures of success are provided by the Task.

* `Sensor`: a generalization of the physical Sensor concept provided by a Simulator, with the capability to provide Task-specific Observation data in a specified format.

* `Observation`: data representing an observation from a Sensor. This can correspond to physical sensors on an Agent (e.g. RGB, depth, semantic segmentation masks, collision sensors) or more abstract sensors such as the current agent state.

Note that the core functionality defines fundamental building blocks such as the API for interacting with the simulator backend, and receiving observations through Sensors. Concrete simulation backends, 3D datasets, and embodied agent baselines are implemented on top of the core API.

## Data
To make things easier we expect `data` folder of particular structure or symlink presented in habitat-lab working directory.

### Scenes datasets
| Scenes models | Extract path | Archive size |
| --- | --- | --- |
| 🆕[HM3D](#HM3D) | `data/scene_datasets/hm3d/{split}/00\d\d\d-{scene}/{scene}.basis.glb` | 130 GB |

#### 🆕Habitat Matterport
Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/089f6a41474f5470ca10222197c23693eef3a001/datasets/HM3D.md):
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival
```

To use an episode dataset provide related config to the Env in [the example](#example) or use the config for [RL agent training](habitat_baselines/README.md#reinforcement-learning-rl).

## Baselines
Habitat Lab includes reinforcement learning (via PPO) and classical SLAM based baselines. For running PPO training on sample data and more details refer [habitat_baselines/README.md](habitat_baselines/README.md).

## Acknowledgments
This is a forked and reduced version of Habitat. If you want to have more information, please visit [the original site](https://github.com/facebookresearch/habitat-lab).

## References
1. 🆕[Habitat 2.0: Training Home Assistants to Rearrange their Habitat](https://arxiv.org/abs/2106.14405) Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra. Advances in Neural Information Processing Systems (NeurIPS), 2021.
2. [Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201). Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, Dhruv Batra. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.
3. [Habitat Challenge 2022](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022). Karmesh Yadav and Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Oleksandr Maksymets and Rishabh Jain and Ram Ramrakhya and Angel X Chang and Alexander Clegg and Manolis Savva and Eric Undersander and Devendra Singh Chaplot and Dhruv Batra. [Online].
