# Learning a Home Vector Representation

Part I rehearsal guide · Project introduction · About 5–6 minutes with pauses to point at the figures.

The page numbers include the title card. Read the **Say** paragraphs aloud on the first rehearsal. On the next pass, try presenting from the bold cues and transitions alone. The backup notes are for questions.

## Page 1 · Title card

**Say.** “Last time, I talked about how a fly can represent the direction it is traveling. This project picks up a question that work left open for me. How could a circuit use those movement signals over time to keep track of home? Today I’ll introduce the anatomy, the model, and the tests I designed for it.”

**Transition.** “First, here’s what I mean by a home vector.”

## Page 2 · What does it mean to keep track of home?

**Cue.** Trace the dashed route, then point to the orange arrowhead at home.

**Say.** “Imagine a fly takes this winding route. At its current location, the orange arrow tells us which way home is and how far away it is in a straight line. As the fly moves, that arrow changes. Keeping it up to date by combining movements over time is what I mean by path integration.”

**Transition.** “Where in the fly brain do the movement signals for this computation come together?”

**Backup.** The simulation keeps home fixed at the starting location. The target is starting position minus current position. A closed route should bring that vector back to zero through cancellation of net displacement.

## Page 3 · Movement signals in the central complex

**Cue.** Point to the ellipsoid body and protocerebral bridge first, then the noduli and fan-shaped body.

**Say.** “This is a close-up of the central complex, near the middle of the fly brain. The ellipsoid body is part of the compass circuit. In the protocerebral bridge, compass neurons pass heading information to neurons that project to the fan-shaped body. Some of those same neurons receive movement-related input in the noduli. The fan-shaped body is where these signals come together in the pathway I’ll focus on.”

**Transition.** “A sideways walk shows why knowing which way the fly faces gives us only part of the picture.”

**Backup.** EPG compass neurons and PFNs both have branches in the protocerebral bridge and can connect directly there. PFNs also receive heading-related input through local Δ7 neurons. The Hulse figure is an anatomical reference from the hemibrain study. The wiring used in this project comes from MaleCNS v1.0. [Hulse et al., anatomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9477501/#fig1) · [Heading inputs to PFNs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9477501/#fig68)

## Page 4 · From facing direction to travel direction

**Cue.** Point to the two orange arrows before naming the PFNs.

**Say.** “The fly on the left faces west and walks to its own right. The fly on the right faces north and walks forward. Both travel north, as the orange arrows show. PFN stands for protocerebral bridge, fan-shaped body, and noduli. PFNd and PFNv combine heading with movement relative to the body. In the diagram on the right, PFNs from both sides converge onto hDeltaB. Inputs to a given hDeltaB neuron tend to favor similar directions of travel in the world, even when they respond to different combinations of heading and body-relative movement.”

**Transition.** “That gives us a signal about movement right now. To track home, a circuit has to combine those movements over time. Here’s the circuit I chose to test that.”

**Backup.** In the published PFN model, the location of the activity peak tracks heading and its amplitude depends on movement along a preferred body-relative direction. The earlier feedforward work reproduces the coordinate transformation. The new task adds a requirement to maintain displacement over time. [Lu et al., PFN tuning and hDeltaB travel signals](https://doi.org/10.1038/s41586-021-04191-x)

## Page 5 · Which neurons are included in the model?

**Cue.** Trace the orange PFN inputs, then point to hDeltaB, the vDeltaE–hDeltaG pair, and hDeltaC.

**Say.** “The orange box holds the 60 PFN inputs. Their activity is prescribed from the fly’s heading and movement. In blue are the four populations whose activity evolves in the model. hDeltaB links this model back to the travel-direction work. vDeltaE and hDeltaG are a candidate circuit for integrating recent movement. hDeltaC exchanges connections with each of the other three populations, so including it lets me ask what it contributes to home tracking. The arrows summarize the population-level connections included from MaleCNS.”

**Transition.** “That’s the wiring. The next step is to turn it into activity that evolves as the fly walks.”

**Backup.** The diagram groups PFNd and PFNv because both project to the same three target populations in this model. A loop indicates connections within a population. The circuit has 65 dynamic neurons across the four groups. The archived graph covers a defined subset of MaleCNS. For example, direct PFN-to-hDeltaG connections were outside the extraction. Janke’s thesis motivates the vDeltaE–hDeltaG pair through recordings and perturbations. hDeltaC’s role in home-vector tracking is a question for the model. [Model setup and scope](cx_home_vector_presentation_brief.md#the-current-setup-one-slide) · [hDeltaC wiring](cx_home_vector_hdeltac_iteration.md#anatomy-provenance) · [Janke thesis](https://doi.org/10.48496/947v-bw85) · [MaleCNS](https://doi.org/10.1016/j.cell.2026.08.015)

## Page 6 · How the model learns to track home

**Cue.** Follow the four boxes. Point to the feedback loop, then to the two vectors below the readout.

**Say.** “At each time step, heading and body-centered velocity determine the PFN inputs. Those inputs and the activity already in the circuit shape its next state. The decoder takes a snapshot of that activity and estimates the two components of the home vector. During training, I compare those predictions with the correct vectors along the route. Training adjusts the allowed connection strengths, the cell-type activity parameters, and the decoder together. The PFN tuning and available connections are fixed.”

**Transition.** “Once the model is trained, how can I tell whether it can track displacement on a new route and keep doing so over time?”

**Backup.** The correct vector is calculated outside the neural model. The decoder reads only the current state, so earlier movement has to influence that state to affect the estimate. Training propagates prediction errors back through the trajectory. In evaluation the parameters are frozen and the activity still evolves. The decoder intercept has one value for each output component. [Training and paired-run details](cx_home_vector_presentation_brief.md#appendix-a1--how-a-trajectory-trains-the-circuit-and-decoder)

## Page 7 · Three tests of the learned home vector

**Cue.** Walk through the panels left to right. Pause briefly after each question.

**Say.** “First, I give the early model a circle it never trained on, then repeat the same lap. Each return brings the fly back to the same place. Does the estimate of home shift anyway? Second, I ask what might limit that drift. Janke observed that hDeltaG activity drops after sugar consumption. That motivates a sugar-suppression test, alongside asking what the standard circuit can learn from training on repeated returns. Finally, one simulated fly waits while another walks. Both then make the same movements. Their correct home vectors stay separated by the displacement from that first walk. Does the circuit preserve that difference?”

**Close.** “Next time I’ll show how the model did on those three tests, and what the results suggest about the movement history it can retain.”

**Backup.** The first circle is a separate evaluation probe. The early training set contained walks followed by pauses, randomized movement sequences, and long straight walks. In the paired comparison, both variants later trained on the same expanded movement dataset. The sugar arm received a sensory event at eligible home-patch reentries. Suppression acts on hDeltaG activity, and the original home anchor stays fixed. The common-input memory test changes the earlier history while holding the later movement fixed. [Training routes](cx_home_vector_presentation_brief.md#appendix-a3--how-the-training-routes-changed) · [Paired training](cx_home_vector_reliable_return_training.md) · [Memory-test design](cx_home_vector_displacement_retention.md#common-input-probe-design)
