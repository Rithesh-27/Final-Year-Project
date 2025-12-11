🚁 Drone RL Project (AirSim + Python)

This project uses AirSim and Reinforcement Learning to train a drone in a simulated environment.
Follow the instructions below to set up the environment correctly.

✅ 1. Install Python 3.10.2 & Create Virtual Environment

The project requires Python 3.10.2 (not higher or lower).

Install Python 3.10.2

Download Python 3.10.2 from the official archive:

🔗 https://www.python.org/downloads/release/python-3102/

Make sure to check:

Add Python to PATH

Install all optional dependencies

Create virtual environment
python3.10 -m venv airsim_env


Activate the environment:

Windows
airsim_env\Scripts\activate

Linux / macOS
source airsim_env/bin/activate


Install dependencies:

pip install -r requirements.txt

✅ 2. Install AirSim and Download Blocks Environment

This project requires the Blocks environment for AirSim.

Download AirSim Blocks binaries

🔗 https://github.com/microsoft/AirSim/releases

Download:

Blocks.zip (under Assets)

Extract it anywhere you like

Inside the extracted folder, you will find:

Blocks.exe

Run the Blocks environment

Simply double-click:

Blocks.exe


This will open the AirSim simulation window.
Keep this running while executing your Python training script.

✅ 3. Configure AirSim Settings.json

AirSim requires a configuration file located in your Documents folder.
Create the following directory (if not present):

Documents/AirSim/


Place your settings.json inside it:

Documents/AirSim/settings.json


AirSim automatically loads this file when you start Blocks.exe.

🚀 Running the Project

Once the virtual environment is activated and Blocks is running:

python main.py


(or whichever script your project uses)

4. Training the drone
Run the file train.py to start training. Blocks.exe should be opened before running. It will display the drone training
Custom actor and critic networks are defined in custom_networks.py
The policy for training ddpg is defined in custom_policy.py
Environment defined in airsim_drone_env.py

