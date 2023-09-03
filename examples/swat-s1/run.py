"""
swat-s1 run.py
"""

from mininet.net import Mininet
from mininet.cli import CLI
from mininet.node import OVSController
from minicps.mcps import MiniCPS

from topo import SwatTopo

import sys


class SwatS1CPS(MiniCPS):

    """Main container used to run the simulation."""

    def __init__(self, name, net):

        self.name = name
        self.net = net

        net.start()

        # start devices
        plc1, plc2, plc3, s1, attacker, defender = self.net.get(
            'plc1', 'plc2', 'plc3', 's1', 'attacker', 'defender')

        attacker.cmd('ip route add 192.168.1.0/24 dev attacker-eth0')
        plc1.cmd('ip route add 10.1.0.0/24 dev plc1-eth0')
        plc2.cmd('ip route add 10.1.0.0/24 dev plc2-eth0')
        plc3.cmd('ip route add 10.1.0.0/24 dev plc3-eth0')
        defender.cmd('ip route add 10.1.0.0/24 dev defender-eth0')

        net.pingAll()

        # SPHINX_SWAT_TUTORIAL RUN(
        plc2.cmd(sys.executable + ' -u ' +' plc2.py &> logs/plc2.log &')
        plc3.cmd(sys.executable + ' -u ' + ' plc3.py  &> logs/plc3.log &')
        plc1.cmd(sys.executable + ' -u ' + ' plc1.py  &> logs/plc1.log &')
        s1.cmd(sys.executable + ' -u ' + ' physical_process.py  &> logs/process.log &')
        # SPHINX_SWAT_TUTORIAL RUN)
        CLI(self.net)

        net.stop()

if __name__ == "__main__":

    topo = SwatTopo()
    net = Mininet(topo=topo, controller=OVSController)

    swat_s1_cps = SwatS1CPS(
        name='swat_s1',
        net=net)
