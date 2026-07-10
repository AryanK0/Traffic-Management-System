import math

vtype_distributions = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- ============================================== -->
    <!-- DRIVER PSYCHOLOGY & KINEMATICS                 -->
    <!-- Sublane model enabled, arbitrary alignments    -->
    <!-- ============================================== -->
    <vTypeDistribution id="2_wheeler">
        <vType id="2w_aggr" probability="0.4" vClass="motorcycle" length="2.0" width="0.8" accel="3.0" decel="4.5" sigma="0.9" tau="0.5" minGap="0.5" minGapLat="0.1" speedFactor="1.2" lcModel="SL2015" lcPushy="1.0" lcAssertive="1.0" latAlignment="arbitrary" color="255,50,50"/>
        <vType id="2w_std"  probability="0.4" vClass="motorcycle" length="2.0" width="0.8" accel="2.5" decel="4.5" sigma="0.7" tau="1.0" minGap="1.0" minGapLat="0.1" speedFactor="1.0" lcModel="SL2015" latAlignment="compact" color="255,100,100"/>
        <vType id="2w_caut" probability="0.2" vClass="motorcycle" length="2.0" width="0.8" accel="2.0" decel="4.5" sigma="0.5" tau="1.5" minGap="2.5" minGapLat="0.2" speedFactor="0.8" lcModel="SL2015" lcPushy="0.0" latAlignment="center" color="255,150,150"/>
    </vTypeDistribution>

    <vTypeDistribution id="3_wheeler">
        <vType id="3w_aggr" probability="0.4" vClass="passenger" length="2.8" width="1.4" accel="2.0" decel="3.5" sigma="0.9" tau="0.5" minGap="0.5" minGapLat="0.2" speedFactor="1.2" maxSpeed="15.0" lcModel="SL2015" lcPushy="1.0" lcAssertive="1.0" latAlignment="arbitrary" color="50,255,50"/>
        <vType id="3w_std"  probability="0.4" vClass="passenger" length="2.8" width="1.4" accel="1.5" decel="3.0" sigma="0.7" tau="1.0" minGap="1.0" minGapLat="0.2" speedFactor="1.0" maxSpeed="15.0" lcModel="SL2015" latAlignment="compact" color="100,255,100"/>
        <vType id="3w_caut" probability="0.2" vClass="passenger" length="2.8" width="1.4" accel="1.0" decel="2.5" sigma="0.5" tau="1.5" minGap="2.5" minGapLat="0.3" speedFactor="0.8" maxSpeed="15.0" lcModel="SL2015" lcPushy="0.0" latAlignment="center" color="150,255,150"/>
    </vTypeDistribution>

    <vTypeDistribution id="4_wheeler">
        <vType id="4w_aggr" probability="0.4" vClass="passenger" length="4.5" width="1.8" accel="2.5" decel="4.5" sigma="0.9" tau="0.5" minGap="0.5" minGapLat="0.3" speedFactor="1.2" lcModel="SL2015" lcPushy="1.0" lcAssertive="1.0" latAlignment="arbitrary" color="50,50,255"/>
        <vType id="4w_std"  probability="0.4" vClass="passenger" length="4.5" width="1.8" accel="2.0" decel="4.0" sigma="0.7" tau="1.0" minGap="1.0" minGapLat="0.4" speedFactor="1.0" lcModel="SL2015" latAlignment="compact" color="100,100,255"/>
        <vType id="4w_caut" probability="0.2" vClass="passenger" length="4.5" width="1.8" accel="1.5" decel="3.5" sigma="0.5" tau="1.5" minGap="2.5" minGapLat="0.5" speedFactor="0.8" lcModel="SL2015" lcPushy="0.0" latAlignment="center" color="150,150,255"/>
    </vTypeDistribution>

    <vTypeDistribution id="heavy_vehicle">
        <vType id="hv_aggr" probability="0.4" vClass="bus" length="10.0" width="2.5" accel="1.2" decel="3.0" sigma="0.9" tau="1.0" minGap="1.0" minGapLat="0.5" speedFactor="1.1" maxSpeed="15.0" lcModel="SL2015" lcPushy="0.5" lcAssertive="0.5" latAlignment="compact" color="255,255,50"/>
        <vType id="hv_std"  probability="0.4" vClass="bus" length="10.0" width="2.5" accel="1.0" decel="2.5" sigma="0.7" tau="1.5" minGap="2.0" minGapLat="0.5" speedFactor="1.0" maxSpeed="15.0" lcModel="SL2015" latAlignment="center" color="255,255,100"/>
        <vType id="hv_caut" probability="0.2" vClass="bus" length="10.0" width="2.5" accel="0.8" decel="2.0" sigma="0.5" tau="2.0" minGap="4.0" minGapLat="0.8" speedFactor="0.8" maxSpeed="15.0" lcModel="SL2015" lcPushy="0.0" latAlignment="center" color="255,255,150"/>
    </vTypeDistribution>

"""

turn_probs = {
    "2_wheeler": {"str": 0.45, "lft": 0.30, "rgt": 0.25},
    "3_wheeler": {"str": 0.45, "lft": 0.30, "rgt": 0.25},
    "4_wheeler": {"str": 0.60, "lft": 0.20, "rgt": 0.20},
    "heavy_vehicle": {"str": 0.85, "lft": 0.10, "rgt": 0.05}
}

base_vph = {
    "2_wheeler": 600,
    "3_wheeler": 300,
    "4_wheeler": 400,
    "heavy_vehicle": 100
}

directions = {
    "n": {"in": "n_in", "str": "s_out", "lft": "e_out", "rgt": "w_out"},
    "s": {"in": "s_in", "str": "n_out", "lft": "w_out", "rgt": "e_out"},
    "e": {"in": "e_in", "str": "w_out", "lft": "s_out", "rgt": "n_out"},
    "w": {"in": "w_in", "str": "e_out", "lft": "n_out", "rgt": "s_out"}
}

surges = [
    {"dir": "n", "vtype": "2_wheeler", "b": 100, "e": 300, "mult": 5.0},
    {"dir": "s", "vtype": "3_wheeler", "b": 1000, "e": 1200, "mult": 4.0},
    {"dir": "e", "vtype": "4_wheeler", "b": 2000, "e": 2200, "mult": 3.0},
    {"dir": "w", "vtype": "2_wheeler", "b": 3000, "e": 3200, "mult": 4.0}
]

xml = vtype_distributions

# 1. Base Flows (0 - 3600s)
xml += '    <!-- ============================================== -->\n'
xml += '    <!-- BASE BACKGROUND FLOWS (Non-Uniform Poisson)    -->\n'
xml += '    <!-- ============================================== -->\n'

for d_name, d_map in directions.items():
    xml += f'    <!-- {d_name.upper()} Incoming Background -->\n'
    for vt, counts in base_vph.items():
        for turn_name, prob in turn_probs[vt].items():
            out_edge = d_map[turn_name]
            vph = int(counts * prob)
            flow_id = f"base_{d_name}_{vt}_{turn_name}"
            xml += f'    <flow id="{flow_id}" type="{vt}" begin="0" end="3600" vehsPerHour="{vph}" from="{d_map["in"]}" to="{out_edge}" departLane="best" departPos="random_free"/>\n'
    xml += '\n'

# 2. Surge Flows (Platoons)
xml += '    <!-- ============================================== -->\n'
xml += '    <!-- CHAOTIC SURGE PLATOONS                         -->\n'
xml += '    <!-- ============================================== -->\n'

for i, s in enumerate(surges):
    d_name = s["dir"]
    d_map = directions[d_name]
    vt = s["vtype"]
    xml += f'    <!-- Platoon {i+1}: {vt} from {d_name.upper()} -->\n'
    for turn_name, prob in turn_probs[vt].items():
        out_edge = d_map[turn_name]
        vph = int(base_vph[vt] * s["mult"] * prob)
        flow_id = f"surge_{i}_{d_name}_{vt}_{turn_name}"
        xml += f'    <flow id="{flow_id}" type="{vt}" begin="{s["b"]}" end="{s["e"]}" vehsPerHour="{vph}" from="{d_map["in"]}" to="{out_edge}" departLane="best" departPos="random_free"/>\n'
    xml += '\n'

xml += "</routes>\n"

with open("C:/Users/FRIDAY/OneDrive/Desktop/Traffic Management System/sumo_env/routes.rou.xml", "w") as f:
    f.write(xml)

print("Generated routes.rou.xml successfully.")
