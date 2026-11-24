# Learning to control a robot arm in MuJoCo: imitation learning, RL and camera perception

I built a 2-joint robot arm environment in **MuJoCo** from scratch, then taught it to reach targets with **imitation learning** (behavior cloning and DAgger) and **reinforcement learning** (SAC). Finally I replaced the simulator's "free" target position with a **camera pipeline** (detection → camera intrinsics → camera-to-world frame transform) and ran the learned policy from vision alone.

https://github.com/USERNAME/REPO/assets/dagger_policy.mp4
<!-- drag dagger_policy.mp4 into the README editor on GitHub to embed it -->

*The DAgger policy reaching targets, using only the tilted camera to find the target.*

![Results](results.png)

## 1. The environment (built myself)
- **Arm:** 2 hinge joints (shoulder and elbow), links 0.30 m and 0.25 m, joint damping, two torque motors limited to ±2 N·m. Physics at 100 Hz, control at 50 Hz.
- **Sensors:** joint position and joint velocity sensors (like encoders).
- **Camera:** an overhead camera, later tilted.
- **Task:** a red target is placed at a random reachable spot (0.15–0.50 m from the base) and the arm starts at a random pose. Success means the hand is within **2 cm** of the target within 100 steps.
- **Observation:** cos and sin of both joint angles, both joint velocities, target x and y (8 values). **Action:** 2 joint torques.

## 2. Expert: inverse kinematics + PD control
An analytic inverse-kinematics solution (law of cosines) gives the joint angles that put the hand on the target. A PD controller on each joint drives the arm there (`torque = Kp·angle error − Kd·velocity`, with the angle error wrapped to the short way around).
**Result: 100/100 success, 38.9 steps on average (≈0.8 s).** This expert generates the demonstrations.

## 3. Imitation learning
### Behavior cloning: data ablation
An MLP (8 → 128 → 128 → 2) trained with MSE to copy the expert's torques. Tested on 100 new random targets:

| Demonstrations | Samples | Training loss | Success |
|---|---|---|---|
| 10 | 339 | 0.0029 | **7%** |
| 50 | 1,978 | 0.0036 | **31%** |
| 200 | 7,858 | 0.0055 | **89%** |

**Lesson: training loss is not the metric.** The 10-demo network has the *lowest* loss and the *worst* robot. In closed loop, its small errors take the arm into states the expert never visited, and the errors compound. More demos cover more of the state space.

### DAgger: fixing compounding errors
Starting from 50 demos, the **network drives** the arm while the **expert labels** every state it visits. Those labels are added to the dataset and the network is retrained:

| Round | Samples | Success |
|---|---|---|
| start (50 demos) | 1,978 | ≈31% |
| 1 | 6,201 | 59% |
| 2 | 10,114 | 75% |
| 3 | 13,257 | 93% |
| 4 | 16,142 | **96%** |

DAgger teaches the network to recover from *its own* mistakes. It ends up close to the expert and above plain behavior cloning with 200 demos (89%).

## 4. Reinforcement learning (SAC)
I wrapped the environment as a standard Gymnasium environment and trained **SAC** (Stable-Baselines3) with reward = −(distance to target). No expert, no demos.

| Env steps | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k |
|---|---|---|---|---|---|---|---|---|
| SAC | 0% | 14% | 12% | 22% | 14% | 22% | 26% | 24% |
| SAC + error vector in observation | 0% | 6% | 8% | 24% | 16% | 22% | 24% | **28%** (avg 6.0 cm) |

**When it wouldn't learn:** the arm learned to get *near* the target (from 35 cm to about 6 cm on average) but rarely stopped within 2 cm. My diagnosis: the reward −distance barely changes between 3 cm and 2 cm, so there is almost no signal for precise stopping. **What I tried:** giving the policy the error vector (target − hand) directly, the same quantity my PD controller uses. It helped a little (24% → 28%, about 6 cm average error), but it didn't solve precision within 40k steps.
**Takeaway:** with a weak reward, RL is far less sample-efficient than demonstrations for this task. A practical next step is to combine them: pretrain with BC or DAgger, then fine-tune with RL, or put the demos into the RL replay buffer.

## 5. Perception: from camera pixels to world coordinates
Instead of reading the target position from the simulator, the robot **finds it with the camera**:
1. **Detection:** threshold red pixels and take the blob centroid (u, v).
2. **Intrinsics:** focal length in pixels from the vertical field of view, `f = (H/2) / tan(fovy/2)`. This turns a pixel into a ray in the camera frame (MuJoCo cameras look along −z).
3. **Extrinsics / frame transform:** rotate the ray into the world frame with the camera's rotation matrix, starting from the camera position.
4. **Ray–plane intersection** with the table plane at the target's height gives the world (x, y).
5. The estimate replaces the true target in the DAgger policy's input.

| Camera | Mean localization error | Success using the camera |
|---|---|---|
| Straight down (1.5 m) | 2.3 mm | 96/100 |
| **Tilted** (moved and rotated) | 6.2 mm | **97/100** |

The tilted camera worked **with no change to the vision code**, because the pipeline uses the camera's full pose. The error grows when tilted because the camera sees the sphere from the side (the blob centroid is no longer the sphere's centre) and each pixel covers more of the table. At ~2 mm per pixel, pixel quantization also limits precision. 6 mm is well inside the 2 cm tolerance, so success didn't drop.

## What I'd do next
- **Combine IL and RL** (BC pretraining plus RL fine-tuning) to get both precision and robustness.
- **Sensor realism:** encoder noise, latency and a state estimator, plus domain randomization (masses, damping, motor strength) for sim-to-real.
- **End-to-end vision:** train the policy directly on camera images instead of a hand-written detector.
- **A harder task:** a 3D arm with a gripper (pick and place), where demonstrations are multimodal.

## Run it
Open `mujoco_arm.ipynb` in Google Colab (**Runtime → T4 GPU**, needed for camera rendering) and run all cells.
Stack: MuJoCo, PyTorch, Stable-Baselines3, Gymnasium, NumPy, mediapy.

*Built in a day as a learning project. I designed and ran every experiment myself, with step-by-step guidance from an AI assistant.*
