import os, re, shutil
from pathlib import Path

file_map = {
    "src/dead_reckon.py": ("src/core/dr.py", "# When the sensors die, we just guess where we are based on where we were. It works... for a few seconds."),
    "src/ekf3_blender.py": ("src/fusion/blender.py", "# Taking our math guesses and ArduPilot's guesses and tossing them in a blender."),
    "src/optical_flow_ins.py": ("src/fusion/opt_flow.py", "# Looking at the ground to figure out how fast we're moving. Literally just staring at dirt."),
    "src/uwb_fusion.py": ("src/fusion/uwb.py", "# Radio beacons beep-booping to give us position. Kinda like GPS but indoors and way more annoying to set up."),
    "src/vio_pipeline.py": ("src/fusion/vio.py", "# Camera plus math equals position. Don't cover the lens."),
    "src/ros2_interface.py": ("src/interfaces/ros2.py", "# Talking to ROS2 because someone decided we need more middleware."),
    "src/slam_interface.py": ("src/interfaces/slam.py", "# Simultaneous Localization and... yeah you know what it means."),
    "src/allan_variance.py": ("src/utils/allan.py", "# Staring at sensor noise for 5 hours to draw a pretty graph."),
    "src/config_loader.py": ("src/utils/config.py", "# Reading the JSON files so we don't hardcode everything (even though we want to)."),
    "src/ekf_comparison.py": ("src/utils/ekf_cmp.py", "# Our ESKF vs ArduPilot's EKF3. Spoiler: ours is better."),
    "src/ground_truth_eval.py": ("src/utils/gt_eval.py", "# Did we actually fly where we thought we did? Let's check."),
    "src/log_replay.py": ("src/utils/replay.py", "# Time travel. Replaying old flights to see why we crashed."),
    "src/time_sync.py": ("src/utils/sync.py", "# Clocks drifting? Not on my watch."),
    "src/fault_manager.py": ("src/safety/fault.py", "# The panic button coordinator."),
    "src/loop_monitor.py": ("src/safety/loop.py", "# Making sure the while loop doesn't take a nap."),
    "src/safety_monitor.py": ("src/safety/safety.py", "# If things get too crazy, we trigger RTL (Return To Land). Please don't hit a tree on the way back."),
    "src/ins_logger.py": ("src/logger/logger.py", "# Writing numbers to CSV really, really fast."),
    "src/structured_logger.py": ("src/logger/struct_log.py", "# Writing JSON logs for the fancy dashboards.")
}

imports_map = [
    (r'from\s+dead_reckon\s+import', 'from core.dr import'),
    (r'from\s+ekf3_blender\s+import', 'from fusion.blender import'),
    (r'from\s+optical_flow_ins\s+import', 'from fusion.opt_flow import'),
    (r'from\s+fusion\.optical_flow_ins\s+import', 'from fusion.opt_flow import'),
    (r'from\s+uwb_fusion\s+import', 'from fusion.uwb import'),
    (r'from\s+vio_pipeline\s+import', 'from fusion.vio import'),
    (r'from\s+ros2_interface\s+import', 'from interfaces.ros2 import'),
    (r'from\s+slam_interface\s+import', 'from interfaces.slam import'),
    (r'from\s+allan_variance\s+import', 'from utils.allan import'),
    (r'from\s+config_loader\s+import', 'from utils.config import'),
    (r'from\s+ekf_comparison\s+import', 'from utils.ekf_cmp import'),
    (r'from\s+ground_truth_eval\s+import', 'from utils.gt_eval import'),
    (r'from\s+log_replay\s+import', 'from utils.replay import'),
    (r'from\s+time_sync\s+import', 'from utils.sync import'),
    (r'from\s+utils\.time_sync\s+import', 'from utils.sync import'),
    (r'from\s+fault_manager\s+import', 'from safety.fault import'),
    (r'from\s+loop_monitor\s+import', 'from safety.loop import'),
    (r'from\s+safety\.loop_monitor\s+import', 'from safety.loop import'),
    (r'from\s+safety_monitor\s+import', 'from safety.safety import'),
    (r'from\s+safety\.safety_monitor\s+import', 'from safety.safety import'),
    (r'from\s+ins_logger\s+import', 'from logger.logger import'),
    (r'from\s+logger\.ins_logger\s+import', 'from logger.logger import'),
    (r'from\s+structured_logger\s+import', 'from logger.struct_log import'),
    (r'from\s+logger\.structured_logger\s+import', 'from logger.struct_log import'),
    (r'from\s+core\.dead_reckon\s+import', 'from core.dr import')
]

for old_file, (new_file, comment) in file_map.items():
    if not os.path.exists(old_file):
        print(f'Missing {old_file}')
        continue
    with open(old_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Safely replace only the FIRST multiline docstring at the top of the file.
    content = re.sub(r'^(#.*?\n)*(\s*)(\"\"\"[\s\S]*?\"\"\")', r'\g<1>\g<2>' + comment, content, count=1)
    
    for pattern, replacement in imports_map:
        content = re.sub(pattern, replacement, content)
        
    content = re.sub(r'ESKFCore', 'ESKF', content)
    
    os.makedirs(os.path.dirname(new_file), exist_ok=True)
    with open(new_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    os.remove(old_file)
    print(f'Processed {old_file} -> {new_file}')
