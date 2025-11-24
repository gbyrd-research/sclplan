# SCLPlan Simulation Code

**NOTE: SCLPlan has undergone a few name changes over the course of the project. In this code base, it was called CALA.**

Code has been tested on MacOS and Linux.

This repo contains the code for the simulation experiments performed in [Constrained Natural Language Action Planning for Resilient Embodied Systems](https://arxiv.org/abs/2510.06357).

## Setup

Create virtual environment and install dependencies:

```bash
conda create -n sclplan python=3.10
conda activate sclplan
pip install -r requirements.txt
```

### Ensure Ollama is installed on your system

Follow the installation instructions found [here](https://ollama.com/download/mac).

In a terminal, ensure whatever Ollama model you want to run has been pulled to your local system. We will use `llama3.1:8b` for our example.

```bash
ollama pull llama3.1
```

## Running the code

We have provided a simple script that will allow you to run SCLPlan with `llama3.1:8b` on the Alfworld validation set.

```bash
./scripts/run_general_alfworld_low_temp.sh
```

Commands to run our code with other configurations are also shown in this bash script, but are commented out.