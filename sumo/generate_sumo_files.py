import os
import subprocess

sumo_dir = 'c:/Users/FRIDAY/OneDrive/Desktop/Traffic Management System/sumo'

nod_xml = """<nodes>
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="n" x="0.0" y="500.0" type="priority"/>
    <node id="s" x="0.0" y="-500.0" type="priority"/>
    <node id="e" x="500.0" y="0.0" type="priority"/>
    <node id="w" x="-500.0" y="0.0" type="priority"/>
</nodes>"""

edg_xml = """<edges>
    <edge id="n_in" from="n" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="n_out" from="center" to="n" priority="1" numLanes="2" speed="13.89"/>
    <edge id="s_in" from="s" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="s_out" from="center" to="s" priority="1" numLanes="2" speed="13.89"/>
    <edge id="e_in" from="e" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="e_out" from="center" to="e" priority="1" numLanes="2" speed="13.89"/>
    <edge id="w_in" from="w" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="w_out" from="center" to="w" priority="1" numLanes="2" speed="13.89"/>
</edges>"""

rou_xml = """<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15.0"/>
    <route id="n_s" edges="n_in s_out"/>
    <route id="s_n" edges="s_in n_out"/>
    <route id="e_w" edges="e_in w_out"/>
    <route id="w_e" edges="w_in e_out"/>
    
    <flow id="flow_n_s" type="car" route="n_s" begin="0" end="3600" probability="0.1"/>
    <flow id="flow_s_n" type="car" route="s_n" begin="0" end="3600" probability="0.15"/>
    <flow id="flow_e_w" type="car" route="e_w" begin="0" end="3600" probability="0.1"/>
    <flow id="flow_w_e" type="car" route="w_e" begin="0" end="3600" probability="0.15"/>
</routes>"""

cfg_xml = """<configuration>
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>"""

with open(os.path.join(sumo_dir, 'nodes.nod.xml'), 'w') as f: f.write(nod_xml)
with open(os.path.join(sumo_dir, 'edges.edg.xml'), 'w') as f: f.write(edg_xml)
with open(os.path.join(sumo_dir, 'routes.rou.xml'), 'w') as f: f.write(rou_xml)
with open(os.path.join(sumo_dir, 'sumo_config.sumocfg'), 'w') as f: f.write(cfg_xml)

print('Running netconvert...')
subprocess.run(['netconvert', '--node-files', 'nodes.nod.xml', '--edge-files', 'edges.edg.xml', '-o', 'network.net.xml'], cwd=sumo_dir)
print('Done!')
