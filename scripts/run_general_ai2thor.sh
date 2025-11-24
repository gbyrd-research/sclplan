# ---------------------------------------------------------------------------- #
# Gpt-4o-mini
# ---------------------------------------------------------------------------- #

# ReAct
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=false \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o_mini/react

# ZS ReAct + Precondition Verification
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o_mini/cala_no_cp

# CALA
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o_mini/cala

# ---------------------------------------------------------------------------- #
# Gpt-4o
# ---------------------------------------------------------------------------- #

# ReAct
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=false \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o/react

# ZS ReAct + Precondition Verification
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o/cala_no_cp

# CALA
python main_single_agent.py \
    agent=ai2thor_cala \
    env=ai2thor \
    tasks=single_agent_hard_ai2thor \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/ai2thor/general/gpt_4o/cala