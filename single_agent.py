from metadrive import MetaDriveEnv
import tqdm

training_env = MetaDriveEnv(dict(
    num_scenarios=1000,
    start_seed=1000,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True
))


test_env = MetaDriveEnv(dict(
    num_scenarios=200,
    start_seed=0,
    random_lane_width=True,
    random_agent_model=True,
    random_lane_num=True
))

for training_epoch in range(2):
    # training
    training_env.reset()
    print("\nStart fake training epoch {}...".format(training_epoch))
    for _ in range(10):
        # execute 10 step
        training_env.step(training_env.action_space.sample())
    training_env.close()

    # evaluation
    print("Evaluate checkpoint for training epoch {}...\n".format(training_epoch))
    test_env.reset()
    for _ in range(10):
        # execute 10 evaluation step
        test_env.step(test_env.action_space.sample())
    test_env.close()

assert test_env.config is not training_env.config
