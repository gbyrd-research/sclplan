# THIS RUNS THE CONSISTENCY EXPERIMENTS WITH ZERO TEMPERATURE AND A FIXED SEED
# THIS SEEKS TO ELIMINATE AS MUCH STOCHASTICITY IN THE MODELS AS POSSIBLE

# Number of times to run the script
NUM_RUNS=10

# ---------------------------------------------------------------------------- #
# Llama3.2:3b
# ---------------------------------------------------------------------------- #

# run react baseline
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama3b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=false \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama3b/react \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run CALA (No classical planner)
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama3b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama3b/cala_no_cp \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run full CALA
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama3b \
        agent.params.use_classical_planner=true \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama3b/cala \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# ---------------------------------------------------------------------------- #
# Llama3.1:8b
# ---------------------------------------------------------------------------- #

for ((i=1; i<=NUM_RUNS; i++))
do
    # run react baseline
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama8b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=false \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama8b/react \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run CALA (No classical planner)
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama8b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama8b/cala_no_cp \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run full CALA
for ((i=1; i<=NUM_RUNS; i++))
do
python main_single_agent.py \
    experiment_type=consistency \
    consistency_run_idx=$i \
    agent=alfworld_cala \
    env=alfworld_consistency \
    tasks=consistency_alfworld \
    agent.params.llm=llama8b \
    agent.params.use_classical_planner=true \
    agent.params.use_precondition_verification=true \
    agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama8b/cala \
    agent.params.llm_kwargs.temperature=0 \
    agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# ---------------------------------------------------------------------------- #
# Llama3.3:70b
# ---------------------------------------------------------------------------- #

# run react baseline
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama70b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=false \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama70b/react \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run CALA (No classical planner)
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama70b \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama70b/cala_no_cp \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run full CALA
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=llama70b \
        agent.params.use_classical_planner=true \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/llama70b/cala \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# ---------------------------------------------------------------------------- #
# GPT-4O-MINI
# ---------------------------------------------------------------------------- #

# run react baseline
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o-mini \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=false \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o_mini/react \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run CALA (No classical planner)
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o-mini \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o_mini/cala_no_cp \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run full CALA
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o-mini \
        agent.params.use_classical_planner=true \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o_mini/cala \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# ---------------------------------------------------------------------------- #
# GPT-4O
# ---------------------------------------------------------------------------- #

# run react baseline
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=false \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o/react \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run CALA (No classical planner)
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o \
        agent.params.use_classical_planner=false \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o/cala_no_cp \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done

# run full CALA
for ((i=1; i<=NUM_RUNS; i++))
do
    python main_single_agent.py \
        experiment_type=consistency \
        consistency_run_idx=$i \
        agent=alfworld_cala \
        env=alfworld_consistency \
        tasks=consistency_alfworld \
        agent.params.llm=gpt-4o \
        agent.params.use_classical_planner=true \
        agent.params.use_precondition_verification=true \
        agent.params.results_dir=experiments/alfworld/consistency_low_temp/gpt_4o/cala \
        agent.params.llm_kwargs.temperature=0 \
        agent.params.llm_kwargs.top_p=0.0001 \
        agent.params.llm_kwargs.seed=42
done