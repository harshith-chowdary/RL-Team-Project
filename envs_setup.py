import os
import subprocess

def run_conda_command(command):
    try:
        subprocess.check_call(command, shell=True)
        print(f"Command '{command}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running '{command}': {e}")

def setup_metadrive():
    run_conda_command("conda install -c conda-forge gym -y")
    run_conda_command("pip install metadrive")

def setup_mamujoco():
    run_conda_command("conda install -c conda-forge mujoco -y")
    run_conda_command("pip install mujoco-py")

def setup_pettingzoo():
    run_conda_command("conda install -c conda-forge gym -y")
    run_conda_command("pip install pettingzoo[magent]")

def setup_flatland():
    run_conda_command("conda install -c conda-forge flatland-rl -y")
    run_conda_command("pip install flatland-rl")

if __name__ == "__main__":
    print("Setting up all environments using Conda and Pip...\n")
    setup_metadrive()
    print("\n")
    setup_mamujoco()
    print("\n")
    setup_pettingzoo()
    print("\n")
    setup_flatland()
    print("\nSetup completed.")
