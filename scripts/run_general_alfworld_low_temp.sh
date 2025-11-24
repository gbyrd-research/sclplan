# ---------------------------------------------------------------------------- #
# Llama3.2:3b
# ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama3b/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama3b/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama3b \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama3b/cala \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run llm-dp
# python main_llmdp_alfworld.py \
#     --output_dir experiments/alfworld/general_low_temp/llama3b/llmdp \
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
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama8b/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama8b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama8b/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# run full CALA
python main_single_agent.py \
    agent=alfworld_cala \
    env=alfworld \
    tasks=single_agent_alfworld \
    agent.params.llm=llama8b \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/general_low_temp/llama8b/cala \
    agent.params.llm_kwargs.temperature=0 \
    agent.params.llm_kwargs.top_p=0.0001 \
    agent.params.llm_kwargs.seed=42

# # run llm-dp
# python main_llmdp_alfworld.py \
#     --output_dir experiments/alfworld/general_low_temp/llama8b/llmdp \
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
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama70b/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama70b \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama70b/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=llama70b \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/llama70b/cala \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run llm-dp
# python main_llmdp_alfworld.py \
#     --output_dir experiments/alfworld/general_low_temp/llama70b/llmdp \
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
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt-3_5-turbo/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-3.5-turbo \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt-3_5-turbo/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-3.5-turbo \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt-3_5-turbo/cala \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42dd

# # run llm-dp
# python main_llmdp_alfworld.py \
#     --output_dir experiments/alfworld/general_low_temp/gpt-3_5-turbo/llmdp \
#     --llm_model gpt-3.5-turbo

# # ---------------------------------------------------------------------------- #
# # GPT-4O-MINI
# # ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o-mini \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o_mini/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o-mini \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o_mini/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o-mini \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o_mini/cala \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # # run llm-dp
# # python main_llmdp_alfworld.py \
# #     --output_dir experiments/alfworld/general_low_temp/gpt_4o_mini/llmdp \
# #     --llm_model gpt-4o-mini

# # ---------------------------------------------------------------------------- #
# # GPT-4O
# # ---------------------------------------------------------------------------- #

# # run react baseline
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=false \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o/react \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run CALA (No classical planner)
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o \
#     agent.params.use_classical_planner=false \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o/cala_no_cp \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # run full CALA
# python main_single_agent.py \
#     agent=alfworld_cala \
#     env=alfworld \
#     tasks=single_agent_alfworld \
#     agent.params.llm=gpt-4o \
#     agent.params.use_classical_planner=true \
#     agent.params.use_precondition_verification=true \
#     agent.params.results_dir=experiments/alfworld/general_low_temp/gpt_4o/cala \
#     agent.params.llm_kwargs.temperature=0 \
#     agent.params.llm_kwargs.top_p=0.0001 \
#     agent.params.llm_kwargs.seed=42

# # # run llm-dp
# # python main_llmdp_alfworld.py \
# #     --output_dir experiments/alfworld/general_low_temp/gpt_4o/llmdp \
# #     --llm_model gpt-4o