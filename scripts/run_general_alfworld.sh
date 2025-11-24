# # ---------------------------------------------------------------------------- #
# # Llama3.2:3b
# # ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general/llama3b/react

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama3b/cala_no_cp

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama3b/cala

# # run llm-dp
# python main_llmdp_alfworld \
#     --output_dir experiments/alfworld/general/llama3b/llmdp \
#     --llm_model llama3b

# # ---------------------------------------------------------------------------- #
# # Llama3.1:8b
# # ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama8b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general/llama8b/react

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama8b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama8b/cala_no_cp

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama8b \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama8b/cala

# # run llm-dp
# python main_llmdp_alfworld \
#     --output_dir experiments/alfworld/general/llama8b/llmdp \
#     --llm_model llama8b

# # ---------------------------------------------------------------------------- #
# # Llama3.3:70b
# # ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama70b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general/llama70b/react

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama70b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama70b/cala_no_cp

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama70b \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/llama70b/cala

# # run llm-dp
# python main_llmdp_alfworld \
#     --output_dir experiments/alfworld/general/llama70b/llmdp \
#     --llm_model llama70b

# # ----------------------------------------------------------------------------
# # GPT-3.5-Turbo
# # ----------------------------------------------------------------------------

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-3.5-turbo \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general/gpt-3_5-turbo/react

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-3.5-turbo \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/gpt-3_5-turbo/cala_no_cp

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-3.5-turbo \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general/gpt-3_5-turbo/cala

# # run llm-dp
# python main_llmdp_alfworld \
#     --output_dir experiments/alfworld/general/gpt-3_5-turbo/llmdp \
#     --llm_model gpt-3.5-turbo

# ---------------------------------------------------------------------------- #
# GPT-4O-MINI
# ---------------------------------------------------------------------------- #

# run react baseline
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=false \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o_mini/react

# run CALA (No classical planner)
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o_mini/cala_no_cp

# run full CALA
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o-mini \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o_mini/cala

# run llm-dp
python main_llmdp_alfworld \
    --output_dir experiments/alfworld/general/gpt_4o_mini/llmdp \
    --llm_model gpt-4o-mini

# ---------------------------------------------------------------------------- #
# GPT-4O
# ---------------------------------------------------------------------------- #

# run react baseline
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=false \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o/react

# run CALA (No classical planner)
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=false \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o/cala_no_cp

# run full CALA
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=gpt-4o \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/general/gpt_4o/cala

# run llm-dp
python main_llmdp_alfworld \
    --output_dir experiments/alfworld/general/gpt_4o/llmdp \
    --llm_model gpt-4o
