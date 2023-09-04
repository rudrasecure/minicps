#!/bin/bash

# Change to root of git project
cd "$(git rev-parse --show-toplevel)"

# Install system dependencies
sudo apt install make python3-pip python-is-python3 mininet openvswitch-testcontroller
curl -fsSL https://get.docker.com/ | sudo bash

# Ensure mininet finds ovs-testcontroller, it looks for ovs-controller
sudo ln -s /usr/bin/ovs-testcontroller /usr/bin/ovs-controller

# Kill existing instance of ovs-testcontroller, starting the simulation spawns one
sudo service openvswitch-testcontroller stop

# Install python dependencies
sudo pip install -r requirements.txt
sudo pip install -r swat-s1-detection/requirements.txt
sudo python setup.py install

# Initialize swat-s1 DB
make swat-s1-init

cat << '_EOF'

Setup done. Now execute `make swat-s1` to start up the simulation.
You will be dropped to a `mininet >` CLI.

In another terminal, you can run 02-start-detections.sh to set up
network-based and anomaly-based detections.

In the CLI, run `attacker sh` to be dropped into a shell on the
attacker machine.

To execute the attack from the attacker's shell, run:

enip_client -a 192.168.1.10 -p LIT101_H:1=1.5
enip_client -a 192.168.1.10 -p LIT101_L:1=0.9

To watch the results of the attack, read the examples/swat-s1/plc1.log.
You will see lines similar to the following:

WARNING PLC1 - lit101 over HH: 1.57 >= 1.20.

_EOF
