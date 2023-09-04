# SWaT Testbed - Attack & Defense

## Introduction

![SWaT Testbed architecture](https://minicps.readthedocs.io/en/latest/_images/swat-tutorial-subprocess.png)

During normal operating conditions the water flows into a Raw water tank (T101) passing through an open motorized valve MV101.
A flow level sensor FIT101 monitors the flow rate providing a measure in m<sup>3</sup>/h. The tank has a water level indicator LIT101
providing a measure in mm. A pump P101 is able to move the water to the next stage. In our simulation we assume that the pump
is either on or off and that its flow rate is constant and can instantly change value.

The whole subprocess is controlled by three PLCs (Programmable Logic Controllers). PLC1 takes the final decisions with the help of PLC2
and PLC3. The following is a schematic view of subprocess's control strategy:

PLC1 will first:
- Read LIT101
- Compare LIT101 with well defined thresholds
- Take a decision (e.g.: open P101 or close MV101)
- Update its status

The thresholds are EtherNet/IP values `LIT101_H` and `LIT101_L`. The attack can be conducted by modifying the `LIT101_H` setpoint to
above 1.2 (which is the high high setpoint, AKA the maximum capacity of the tank). Setting `LIT101_H` to any value above 1.2 causes
the water level to increase until that point, effectively overflowing the tank.

## Steps to execute:

### Setup phase

- Create a fresh installation of Ubuntu 22.04 Server with access to the internet
- `git clone --recursive https://github.com/rudrasecure/minicps.git`
- `cd minicps`
- `scripts/setup/01-setup-deps.sh`: This will set up dependencies.
- `make swat-s1`: This will land you in a mininet CLI.
- In another terminal, `scripts/setup/02-start-detections.sh`: This starts up Suricata and ML-based detection

### Setting up the attack machine

- In the mininet prompt, run `noecho attacker bash` to run a bash terminal inside the attacker's network namespace.
- To fix the terminal, run `stty sane` and press Enter. Note that you won't be able to see this as you type, so you'll have to blindly type it out.
- After this, you'll be able to run commands (almost) normally.

### Attack phase
In the attacker's shell, run the following commands:

```bash
enip_client -a 192.168.1.10 -p LIT101_H:1=1.5
enip_client -a 192.168.1.10 -p LIT101_L:1=0.9
```

If you switch back to the window that is showing the detection log, you will notice alerts being raised.
