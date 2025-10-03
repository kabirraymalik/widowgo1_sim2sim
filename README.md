# WidowGo1_Sim2Sim

## 📦 Installation

1. Set up conda environment with python >=3.10
   ```
   pip install -r requirements.txt
   ```

### ⚙️ Running a pretrained policy

Test widowGo1 pretrained model:

Additional arguments can be found in play_pt_policy.py
```
python play_pt_policy.py
```
Controls:
space -> pause sim
s (while paused) -> advance 1 physics step
up down left right -> base linvel
q e -> base angvel
numkeys 1-0 -> switch between sampled ee pose commands
v -> toggle visualization
Note: DO NOT PRESS ESC KEY

#### Other

Test widowGo1 xml model with pd control:

```
python pd_control.py
```