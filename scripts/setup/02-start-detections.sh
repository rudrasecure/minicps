#!/bin/bash -e

# Change to root of git project
cd "$(git rev-parse --show-toplevel)"

# Set up traffic mirroring to defender's network interface (s1-eth5)
sudo ovs-vsctl -- --id=@m create mirror name=corespanmirror -- add bridge s1 mirrors @m
sudo ovs-vsctl -- --id=@m get port s1-eth5 -- set mirror corespanmirror select_all=true output-port=@m

sudo docker ps -q --filter name=suricata | grep -q . && docker stop suricata && docker rm -fv suricata
sudo docker run -d --rm --name suricata --net=host --cap-add=net_admin --cap-add=net_raw --cap-add=sys_nice -v $PWD/logs/suricata/:/var/log/suricata -v $PWD/suricata/etc:/etc/suricata jasonish/suricata:latest -i s1-eth5

tmux new-session -d -s defender
tmux send-keys -t defender:0 "tail -F -n 0 $PWD/logs/suricata/fast.log" C-m
tmux split-window -t defender:0 -h
tmux send-keys -t defender:0.1 "cd swat-s1-detection && python wrapper.py" C-m
tmux attach-session -t defender
