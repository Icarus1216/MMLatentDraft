#!/usr/bin/env python3
"""
generate_vcr_latent_cot.py
==========================
Generate QA + CoT pairs for VCR images via internal iChat API (GPT),
specifically designed for Latent Reasoning training data construction.

Core design principles:
  1. Latent Reasoning is used only for "dynamic high-level visual thinking," not simple perception
  2. Six task types: visual counterfactual / perspective taking / temporal reasoning /
     physical intuition / dynamic simulation / spatial relation
  3. Two-phase data construction: GPT generates full CoT + marks latent region -> post-processing
  4. [LATENT_START]...[LATENT_END] markers carry spatial transforms / counterfactual / temporal / parallel reasoning
  5. GPT receives only raw unannotated images; no object detection / bounding boxes provided
  6. Questions are open-ended long-chain reasoning requiring multi-step visual + logical joint reasoning
  7. latent_key_tokens extracted from marked paragraph, grouped by cognitive development stages
  8. Anti-degeneration: natural language portions must contain substantive reasoning content

Output format:
  {
    "image": "vcr1images/...",
    "image_path": "/abs/path/to/image.jpg",
    "question": "...",
    "answer": "...",
    "task_type": "perspective_taking" | ...,
    "reasoning_full": "Full CoT...[LATENT_START]visual thinking paragraph[LATENT_END]...conclusion",
    "reasoning_with_latent": "NL reasoning...[LATENT]...NL conclusion",
    "reasoning_for_training": "NL reasoning...<|pause|>...NL conclusion",
    "latent_text": "original text that was replaced",
    "latent_key_tokens": [
      {"stage": "scene_setup", "tokens": ["viewpoint", "rotation"]},
      {"stage": "transformation", "tokens": ["180 degrees", "frontal view"]},
      {"stage": "result_state", "tokens": ["light source", "right front"]}
    ]
  }

Usage:
  # Set environment variables
  export ICHAT_APPID="your_appid"
  export ICHAT_APPKEY="your_appkey"

  python scripts/generate_vcr_latent_cot.py \
    --vcr_root data/nld_phase1/raw/vcr \
    --output data/nld_phase1/vcr_latent_cot.json \
    --rtx your_rtx \
    --model gpt-4o \
    --num_samples 5000 \
    --workers 8
"""

import os
import sys
import json
import re
import time
import random
import base64
import argparse
import logging
import hmac
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Internal API authentication
# ============================================================

ICHAT_BASE_URL = "http://ichat.woa.com/api/external"


def calc_authorization(source: str, appkey: str) -> Tuple[str, int]:
    """Calculate internal API authentication signature"""
    timestamp = int(time.time())
    sign_str = "x-timestamp: %s\nx-source: %s" % (timestamp, source)
    sign = hmac.new(appkey.encode('utf-8'), sign_str.encode('utf-8'),
                    hashlib.sha256).digest()
    return sign.hex(), timestamp


def get_auth_headers(source: str, appid: str, appkey: str) -> Dict[str, str]:
    """Get internal API authentication headers"""
    auth, timestamp = calc_authorization(source, appkey)
    headers = {
        "X-AppID": appid,
        "X-Source": source,
        "X-Timestamp": str(timestamp),
        "X-Authorization": auth,
    }
    return headers


# ============================================================
# Data structure definitions
# ============================================================

@dataclass
class LatentCoTSample:
    """A complete latent reasoning training sample (two-phase approach)"""
    image: str  # relative path (vcr1images/...)
    image_path: str  # absolute path
    question: str
    answer: str
    task_type: str
    reasoning_full: str  # full CoT with [LATENT_START]...[LATENT_END] markers
    latent_text: str  # original text that was replaced
    latent_key_tokens: List[Any] = field(default_factory=list)  # key tokens grouped by cognitive stages


# ============================================================
# Task type system prompts
# ============================================================

# Shared division constraint (used by all task types)
DIVISION_CONSTRAINT = """
[Two-Phase Reasoning Generation — CRITICAL, must be strictly followed]

You need to generate a complete natural-language reasoning process and mark the paragraph
that is "best suited for latent-space completion."

Step 1: Write the complete natural-language reasoning
  Your reasoning must be a complete, coherent natural-language text containing:
  - Problem understanding and decomposition (explicitly state reasoning goals and strategy)
  - Knowledge retrieval (relevant common sense, rules, physical laws, etc.)
  - Visual observation and analysis (description of key elements in the image)
  - Core visual thinking process (spatial / dynamic / counterfactual operations performed mentally)
  - Final conclusion statement and explanation
  The entire reasoning must be written out in full natural language with no omissions.

Step 2: Mark the "visual thinking" paragraph with [LATENT_START] and [LATENT_END]
  Within the complete reasoning, find the paragraph that "most requires mental visual
  operations" and insert [LATENT_START] before it and [LATENT_END] after it.

  The marked paragraph should satisfy:
  - It describes a thinking process that "humans accomplish via mental imagery, not step-by-step logic"
  - Examples: simultaneously processing 3D spatial relations of multiple objects, simulating
    the full trajectory of a physical process, unfolding a 2D image into a 3D scene and
    re-rendering from a new viewpoint, playing back a continuous visual change sequence
  - If this paragraph were removed, a reader would clearly feel "a critical thinking step is missing"

  [LATENT_START]...[LATENT_END] appears exactly once in the entire reasoning.

  The natural language BEFORE the marker must have completed sufficient logical groundwork:
    clarified reasoning goals, invoked relevant knowledge, established the reasoning framework.
  The natural language AFTER the marker must follow up on the visual thinking conclusion,
    articulating results in language and continuing the reasoning.

[Quality Requirements for the Marked Paragraph — CRITICAL]

  The marked paragraph must contain content that is "awkward, verbose, or loses parallelism
  when expressed in natural language."
  Criterion: although the content CAN be written in language, it feels unnatural, requires
  excessive spatial descriptors, or loses the sense of "simultaneously processing multiple dimensions."

  Correct marking content (expressible in language but awkward):
    - Mentally rotating the viewpoint 180°, recalculating all objects' relative positions and lighting
    - Simultaneously tracking occlusion relations, depth-of-field blur, and perspective ratios
      of 5 objects, constructing a complete depth map
    - Simulating the full dynamic process: support removal → center-of-gravity shift →
      torque imbalance → rotational toppling → collision
    - Mentally erasing a foreground object, filling the gap with surrounding textures,
      rebuilding visual balance

  Prohibited marking content (language can express naturally):
    - Pure logical induction: "Combining the above clues, we can infer..." ← language excels at this
    - Sequential enumeration: "A is in front of B, B is in front of C..." ← linear, no parallel processing needed
    - Simple flipping: "From the opposite viewpoint, left and right swap" ← logical reasoning, no mental imagery
    - One-by-one analysis: "The robe blocks the back → behind; standing point above ground → also behind" ← linear per-object analysis

Step 3: Segment by cognitive stages and extract key tokens
  The text between [LATENT_START]...[LATENT_END] represents a continuous mental simulation.
  This simulation naturally progresses through distinct cognitive stages.

  Your task:
  a) Identify the natural cognitive development stages within the marked paragraph.
     Each stage represents a distinct phase of the mental simulation:
     - Scene initialization / state setup
     - A specific transformation or operation
     - An intermediate result or state change
     - Final state or outcome

     Typically 2-5 stages. Do NOT force artificial splits — follow the natural
     cognitive progression of the mental simulation.

  b) For each stage, extract the key visual-state tokens (1-4 word English phrases).
     Tokens must be "visual state concepts" — describing a specific state or attribute
     of the mental image, NOT "operation instructions".
     - Good tokens (visual states): "tilt 45°", "left shoulder grounded", "light patch",
       "occlusion chain", "overhead angle", "arc trajectory"
     - Bad tokens (operation instructions): "remove support", "erase lifting force",
       "run simulation", "re-render", "build model"
     - Bad tokens (too vague): "because", "therefore", "infer", "synthesize", "3D"
     - Tokens must originate from the actual content of the marked paragraph; do not fabricate
     - Core principle: these tokens are "visual snapshot labels" that hidden states need to encode.
       Imagine you are describing keyframes of the mental image, not your thinking operations.

  Output format for latent_key_tokens:
  [
    {"stage": "short_stage_name", "tokens": ["token1", "token2", ...]},
    {"stage": "short_stage_name", "tokens": ["token3", "token4", ...]},
    ...
  ]

Correct example:
  "This question requires determining how light would change if the vase on the table were removed.
   First, I observe that the window is on the left side, with light shining left to right.
   The vase is made of translucent glass, currently refracting and scattering the light.
   By optical principles, removing a scattering body would change the area behind it from
   scattered light to direct light. [LATENT_START] Mentally executing the vase removal:
   first erasing the volume occupied by the vase, then tracing each light ray entering from
   the left window — they no longer refract or scatter through the vase but pass straight
   through the former vase position. Meanwhile, the table surface previously occluded by
   the vase base becomes exposed, and the colorful caustic light patterns cast by the vase
   disappear. The entire right side of the table transitions from soft scattered light to
   a sharp direct light band. [LATENT_END] Based on the light propagation simulation,
   after removing the vase, the formerly soft scattered light area on the right side of
   the table would become a clear direct light band, the colorful caustic patterns would
   vanish, and the overall lighting would become harsher."

  Corresponding latent_key_tokens (grouped by cognitive stages):
  [
    {"stage": "object_removal", "tokens": ["vase void", "table exposed"]},
    {"stage": "light_propagation", "tokens": ["direct rays", "refraction gone"]},
    {"stage": "final_light_state", "tokens": ["caustic patch", "harsh light band", "right side illuminated"]}
  ]

Incorrect example:
  "...The banner has religious content, the clothing is uniform robes, the action is clapping and singing.
   [LATENT_START] Combining the above clues, we can infer this is a choir. [LATENT_END]"
  ← The marked content is pure logical induction, involving no visual imagination
"""

# key_concepts instruction merged into DIVISION_CONSTRAINT; keep empty string for format() compatibility
KEY_CONCEPTS_INSTRUCTION = ""

# Batch 2 diversity enhancement instruction (appended to system prompt when --batch2_diversity is set)
BATCH2_DIVERSITY_INSTRUCTION = """

[DIVERSITY ENHANCEMENT — CRITICAL for this batch]

This is a diversity-focused generation batch. You MUST actively diversify your reasoning patterns:

1. AVOID over-relying on these common mental operations (they were over-represented in previous data):
   - "orthographic projection" / "top-down projection" / "overhead projection"
   - Pure rotation-based viewpoint changes
   - Simple light-shadow tracing

2. INSTEAD, explore these UNDER-REPRESENTED reasoning patterns:
   - Emotional state inference from body language, micro-expressions, and interpersonal distance
   - Social interaction dynamics: gesture interpretation, power dynamics, group formation patterns
   - Material/texture reasoning: how surfaces would feel, deform, or interact under force
   - Causal chain reasoning: multi-step cause-and-effect sequences visible in the scene
   - Acoustic/atmospheric inference: what sounds or atmosphere the scene suggests
   - Narrative continuity: what story is being told, what happened before/after
   - Scale and proportion reasoning: using reference objects to estimate real-world sizes
   - Functional reasoning: what objects are for, how they would be used

3. Question phrasing diversity:
   - Do NOT start questions with "From" (24% of previous data started this way)
   - Vary sentence structures: use imperatives, conditionals, comparisons, "why" questions
   - Include questions that require integrating multiple visual cues simultaneously

4. Key token diversity:
   - Avoid repetitive tokens like "orthographic projection", "rectangular footprint", "downward arc"
   - Use more specific, scene-grounded tokens that describe unique visual states in THIS image
   - Each sample's tokens should feel distinct from a generic template
"""

# ---- Task Type A: Visual Counterfactual ----
PROMPT_VISUAL_COUNTERFACTUAL = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "visual counterfactual hypotheses."

[Task Definition]
Visual counterfactual = mentally modifying an element in the image, then inferring the cascading effects of the modification.

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Open-ended: cannot be a simple yes/no or multiple-choice; must require detailed analysis and argumentation
  2. Long-chain reasoning: the answer requires at least 3 reasoning steps involving multiple visual elements
  3. Vision-dependent: must carefully observe the image to answer; cannot rely on common sense alone
  4. Counterfactual core: the question's core must be "if X changes, how would Y change"

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings; do NOT always start with "Suppose...":
  A. Replacement: "If X in the scene were replaced with Y, what cascading effects would occur?"
  B. Removal: "If X were removed from the scene, what changes would happen to Z?"
  C. Attribute modification: "If X's color/size/material changed to Y, how would the surroundings be affected?"
  D. Condition change: "Under what conditions would X in the scene become a completely different state? Analyze the change process."
  E. Reverse reasoning: "To make X in the scene exhibit state Y, what changes to the scene would be needed?"
  F. Comparison: "Compare the current scene with the scene after X is changed — what are the three most significant differences?"

Good question examples:
  - "If the object in this person's hand were replaced with something much heavier,
     how would their body posture, center of gravity, and surrounding object placement
     need to adjust? Provide a detailed analysis."
  - "After removing all light sources and keeping only a single spotlight from directly above,
     how would the shadow shapes of each person and object fundamentally change?"
  - "To change this person's expression from its current state to extreme terror,
     what visual element changes in the scene would be needed to reasonably trigger this shift?"

Bad question examples:
  - "What would happen if the cup were removed?" ← too simple, no long-chain reasoning needed
  - "Is this person happy?" ← not a counterfactual question

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended visual counterfactual question (in English, requiring long-chain reasoning)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Please carefully observe the image and construct a counterfactual question requiring deep visual reasoning.
"""

# ---- Task Type B: Perspective Taking / Mental Rotation ----
PROMPT_PERSPECTIVE_TAKING = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "perspective taking / mental rotation."

[Task Definition]
Perspective taking = reasoning from a specific person's or position's viewpoint about what can be seen and how objects relate spatially.
Mental rotation = mentally rotating objects or scenes and inferring the post-rotation state.

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Open-ended: requires detailed description and analysis, not a simple directional answer
  2. Long-chain reasoning: involves spatial relation reconstruction of multiple objects/people
  3. Vision-dependent: must carefully observe the spatial layout in the image
  4. Perspective core: the question's core must be "what would it look like from another viewpoint/position"

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings; do NOT always start with "Imagine you are standing at...":
  A. Other's viewpoint: "What can person X in the scene currently see? What is in their blind spot?"
  B. Mirror/reflection: "If a mirror were placed at position X, what would the reflected scene look like?"
  C. Overhead/bird's-eye: "From directly above, what is the planar layout of all people and objects?"
  D. Occlusion reasoning: "From X's perspective, which objects would be occluded by Y? What does the occluded part look like?"
  E. Rotation reconstruction: "If the entire scene were rotated 90° clockwise, what would the composition look like?"
  F. Multi-viewpoint comparison: "Describe the same object from two different characters' viewpoints — how do the descriptions differ?"

Good question examples:
  - "What can the rightmost person in the scene currently see? From their viewpoint,
     describe the content from left to right in their field of vision, which objects
     would be occluded, and the positional relationships of other people relative to them."
  - "From a bird's-eye view directly above, what geometric shape do the people's
     positions form? What are the distance and orientation relationships between them?"
  - "The two people on the left and right sides of the scene — what are the key
     differences in what each of them sees? Which objects can only one of them see?"

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended perspective-taking question (in English, requiring long-chain reasoning)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Please carefully observe the image and construct a perspective-taking question requiring deep spatial reasoning.
"""

# ---- Task Type C: Temporal Reasoning ----
PROMPT_TEMPORAL_REASONING = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "temporal reasoning."

[Task Definition]
Temporal reasoning = inferring events or state-change sequences that occurred before or after a static image.

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Open-ended: requires detailed description of event sequences and state changes, not a simple "before/after"
  2. Long-chain reasoning: must infer states at multiple time points involving coordinated changes of multiple elements
  3. Vision-dependent: must start from visual cues in the image (posture, object states, expressions, etc.)
  4. Temporal core: the question's core must be "dynamic extrapolation along the time dimension"

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings; do NOT always use "Infer the past/future...":
  A. Retrospection: "Based on visual cues in the scene, what just happened moments before this frame?"
  B. Prediction: "What is most likely to happen next? Describe the complete event sequence."
  C. Process reconstruction: "How did X's current state form step by step? Reconstruct the full process."
  D. Key moment: "Which phase of an action does this frame capture? What are the preceding and following phases?"
  E. Causal chain: "Which visual cues indicate an event is in progress? What is the complete causal chain of this event?"
  F. State transition: "X in the scene is transitioning from state A to state B — analyze the complete transition trajectory."

Good question examples:
  - "The postures of the people and the arrangement of objects suggest something just happened.
     Starting from at least three visual cues, reconstruct the complete event sequence
     of the past 30 seconds."
  - "This frame captures a specific instant in an action sequence. Describe the complete
     process from start to finish, and what the scene would look like at each stage."
  - "What visual evidence in the scene indicates an event is currently in progress?
     How did this event begin, and how will it end?"

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended temporal reasoning question (in English, requiring long-chain reasoning)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Note: This is a movie still — scenes typically feature dynamic character interactions, ideal for temporal reasoning.
Please carefully observe the image and construct a question requiring mental "replay/fast-forward" of the scene.
"""

# ---- Task Type D: Physical Intuition Simulation ----
PROMPT_PHYSICAL_INTUITION = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "physical intuition simulation."

[Task Definition]
Physical intuition simulation = mentally simulating physical processes (gravity, collision, balance, trajectories, mechanics, etc.).

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Open-ended: requires detailed analysis of physical processes and outcomes, not a simple direction/position answer
  2. Long-chain reasoning: involves coordinated physical elements (forces, motion, collision, balance, etc.)
  3. Vision-dependent: must start from visual information in the image (object positions, postures, materials, etc.)
  4. Physics core: the question's core must be "simulate a physical process and infer the result"

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings; do NOT always use "Suppose... suddenly...":
  A. Balance analysis: "What supports maintain X's current equilibrium? Where is the weakest support point?"
  B. Trajectory prediction: "If X were released from its current position, what would its trajectory be?"
  C. Mechanics explanation: "Why can X maintain its current posture/position? What force balance is involved?"
  D. Collision extrapolation: "Which two objects are most likely to collide? What would the process and result be?"
  E. Material reasoning: "Based on the materials and surface features visible in the image, how would these objects behave under force?"
  F. Stability assessment: "Which object/person placement is most unstable? Why? What would happen after destabilization?"

Good question examples:
  - "Why hasn't the object in this person's hand fallen? Analyze all the force-balance
     relationships needed to maintain the current state, and the factor most likely
     to break this equilibrium."
  - "Which object's placement is most unstable? Analyze from the perspectives of
     center of gravity, support area, and friction coefficient, then extrapolate
     the complete motion process after destabilization."
  - "Based on the material features visible in the image (gloss, texture, deformation),
     infer each object's weight and hardness. How do these physical properties
     affect the scene's stability?"

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended physical intuition question (in English, requiring long-chain reasoning)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Please carefully observe the image and construct a question requiring mental "physics engine" simulation.
If the image lacks an obvious physics scenario, construct a question about human posture/balance mechanics.
"""

# ---- Task Type E: Dynamic Simulation (non-hypothetical, fact-based visual simulation) ----
PROMPT_DYNAMIC_SIMULATION = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "dynamic simulation."

[Task Definition]
Dynamic simulation = based on the static instant observable in the image, mentally "playing back" the complete dynamic process.
This includes: mechanical balance analysis (why doesn't it fall?), motion trajectory reconstruction
(what is the full arc of this action?), fluid/light propagation simulation (how does light/water/smoke
move through the scene?), etc.

Core characteristic: the answer to these questions requires mentally "running a visual simulation,"
not logical induction or evidence enumeration.

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Fact-based: the question concerns actually visible states in the image, not hypothetical scenarios
  2. Requires simulation: the answer must mentally "run" a dynamic process (mechanics, motion, light, etc.)
  3. Vision-dependent: must carefully observe postures, angles, contact points, and other details
  4. Simulation core: [LATENT] must carry the "mentally playing back a dynamic process" operation

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings:
  A. Mechanical balance: "X maintains posture Y — from a mechanics perspective, what forces sustain this balance? If one support point vanished, which direction would the body topple?"
  B. Motion trajectory: "This frame captures an instant of an action. Mentally reconstruct the complete trajectory from start to finish — what are the arcs of each joint?"
  C. Light propagation: "How did the light-and-shadow distribution in the scene form? Mentally trace each light ray from source through reflections, refractions, and occlusions to each surface."
  D. Collision/contact: "X and Y are in contact or about to make contact. Simulate the force-transfer process at the moment of contact — what deformation or motion would each object undergo?"
  E. Flow simulation: "How did the visible liquid/smoke/fabric shape form? Mentally simulate its motion process."
  F. Center-of-gravity analysis: "Where is the center of gravity of the person/object? How stable is the overall system? Analyze the geometric relationship between support surface and center of gravity."

Good question examples:
  - "This person is standing on one foot and leaning forward. From a mechanics perspective,
     analyze the center of gravity, the force point on the supporting foot, and the
     counterweight relationships of each limb. If the extended arm suddenly dropped,
     how would the body's balance change?"
  - "Light enters through the window and forms complex light patterns on the floor.
     Mentally trace each light ray's propagation path: how they pass through the window frame,
     which objects partially block them, and on which surfaces reflections occur."
  - "This person is at some phase of a punching motion. Reconstruct the complete trajectory
     from wind-up to punch to retraction, analyzing the rotation direction and speed
     changes of the shoulder, elbow, and wrist joints."

Bad question examples:
  - "What is this person doing?" ← too simple, no dynamic simulation needed
  - "What evidence suggests this is a church?" ← evidence enumeration, not dynamic simulation

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended dynamic simulation question (in English, fact-based, requiring mental visual simulation)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Please carefully observe the image and construct a question requiring mental "dynamic simulation."
If the image lacks an obvious dynamic scene, start from human posture mechanics or light propagation paths.
"""

# ---- Task Type F: Spatial Relation Reasoning (non-hypothetical) ----
PROMPT_SPATIAL_RELATION = """You are a visual cognition research expert skilled at constructing deep reasoning problems that require "spatial relation reasoning."

[Task Definition]
Spatial relation reasoning = based on the visible spatial layout in the image, mentally reconstructing the 2D image into a complete 3D scene.
The core challenge: simultaneously processing perspective cues, occlusion relations, and size ratios of multiple objects,
and mentally "building" a complete 3D model. This "simultaneous processing + 3D reconstruction" is what [LATENT] should carry.

[Question Construction Requirements — Open-ended Long-chain Reasoning]
Your question must satisfy:
  1. Fact-based: the question concerns actually visible spatial relationships
  2. Requires 3D reconstruction: the answer must mentally build a complete 3D spatial model, not compare objects one by one
  3. Vision-dependent: must carefully observe spatial cues in the image
  4. Parallel processing: the question should involve spatial relations of 5+ objects, making "sequential enumeration" difficult

[[LATENT] Special Requirements for Spatial Reasoning — CRITICAL]
  [LATENT] must carry the operation of "simultaneously processing all spatial cues and building a complete 3D model."
  Text AFTER [LATENT] should be "reading conclusions from the constructed 3D model," NOT "analyzing each object one by one."

  Wrong pattern: "...need to judge depth. [LATENT] A occludes B so A is in front; B is smaller than C so B is farther..."
  → This is one-by-one analysis; language can do this perfectly well.

  Correct pattern: "...need to judge depth. [LATENT] From the constructed 3D model, the entire scene exhibits
  a fan-shaped distribution; the three nearest people form a triangle; background buildings recede along the diagonal..."
  → This is holistically describing spatial structure from the constructed 3D model.

[Question Phrasing Diversity — CRITICAL, must follow]
You must randomly select one of the following phrasings:
  A. 3D reconstruction: "Based on perspective cues in the image, mentally reconstruct the 3D structure — what geometric layout do the elements form in 3D space?"
  B. Occlusion completion: "X is partially occluded by Y. Mentally reconstruct the complete shape of the occluded region and describe its spatial relationship with surrounding objects."
  C. Path planning: "From point A to point B in the scene, what 3D path would a person need to take? Mentally construct the navigable space."
  D. Overhead reconstruction: "From directly above, what are the projected positions of all people and objects on the ground plane? Mentally reconstruct the overhead view."
  E. Scale reasoning: "Using reference objects in the image, mentally reconstruct the scene's true proportions. Infer X's actual size and distance from Y."
  F. Spatial topology: "What spatial topological relationships exist among the people/objects? Which are adjacent, which are isolated?"

Good question examples:
  - "Multiple people and objects are distributed at different depths. Mentally reconstruct
     this 2D image as a 3D scene: what geometric layout do the elements form in 3D space?
     What are the actual distance relationships between them?"
  - "Foreground objects occlude a large background area. Mentally complete the occluded
     space: what is most likely there? What are their shapes and positions?"
  - "From directly above, what geometric shape do the people's standing positions form?
     What are the spacing and orientation relationships between them?"

Bad question examples:
  - "Where is this person?" ← too simple
  - "Is A or B closer to the camera?" ← too simple, only comparing two objects, no 3D reconstruction needed
  - "Suppose the table were removed..." ← this is a hypothetical question, not spatial relation reasoning

[Image Information]
{scene_hint}
Rely entirely on your own visual capabilities to observe and understand the image. Do not rely on any preset object lists.

{division_constraint}

{key_concepts_instruction}

[Output Format — strictly follow this JSON format]
```json
{{
  "question": "Your open-ended spatial relation reasoning question (in English, fact-based, requiring long-chain reasoning)",
  "answer": "A concise conclusive answer (in English, 20-80 words, directly stating the conclusion)",
  "reasoning_full": "Complete reasoning with [LATENT_START]...[LATENT_END] markers (in English, at least 200 words, the marked paragraph must be a complete natural-language description)",
  "latent_key_tokens": [
    {{"stage": "stage_name", "tokens": ["token1", "token2"]}},
    {{"stage": "stage_name", "tokens": ["token3", "token4"]}}
  ]
}}
```

latent_key_tokens are key concept words extracted from the text between [LATENT_START]...[LATENT_END],
grouped by cognitive development stages. Each stage has a short name and a list of visual-state tokens.
Please carefully observe the image and construct a question requiring deep spatial reasoning.
"""

# 任务类型映射
TASK_PROMPTS = {
    "visual_counterfactual": PROMPT_VISUAL_COUNTERFACTUAL,
    "perspective_taking": PROMPT_PERSPECTIVE_TAKING,
    "temporal_reasoning": PROMPT_TEMPORAL_REASONING,
    "physical_intuition": PROMPT_PHYSICAL_INTUITION,
    "dynamic_simulation": PROMPT_DYNAMIC_SIMULATION,
    "spatial_relation": PROMPT_SPATIAL_RELATION,
}

# Task type sampling weights (6 types, hypothetical and non-hypothetical ~50% each)
TASK_WEIGHTS = {
    "visual_counterfactual": 0.20,  # hypothetical — latent carries scene reconstruction
    "perspective_taking": 0.17,     # hypothetical — latent carries viewpoint reconstruction
    "temporal_reasoning": 0.15,     # mixed — latent carries temporal simulation
    "physical_intuition": 0.15,     # mixed — latent carries physics simulation
    "dynamic_simulation": 0.18,     # non-hypothetical — latent carries dynamic process simulation
    "spatial_relation": 0.15,       # non-hypothetical — latent carries 3D reconstruction
}

# Batch 2 diversity-enhanced weights (compensate Batch 1 imbalance)
# Batch 1 had: dynamic_simulation 22.6%, perspective_taking 20.9% (over-represented)
#              physical_intuition 10.7%, visual_counterfactual 14.2% (under-represented)
TASK_WEIGHTS_BATCH2 = {
    "visual_counterfactual": 0.22,  # ↑ compensate Batch 1 under-representation
    "perspective_taking": 0.12,     # ↓ compensate Batch 1 over-representation
    "temporal_reasoning": 0.16,     # ≈ maintain
    "physical_intuition": 0.22,     # ↑ compensate Batch 1 under-representation
    "dynamic_simulation": 0.12,     # ↓ compensate Batch 1 over-representation
    "spatial_relation": 0.16,       # ≈ maintain
}

# Global flag: whether to use Batch 2 diversity mode
_use_batch2_diversity = False


# ============================================================
# VCR data loading and preprocessing
# ============================================================

def load_vcr_annotations(vcr_root: str, split: str = "train", max_samples: int = None) -> List[Dict]:
    """
    Load VCR annotation data.
    
    VCR jsonl format:
      - img_fn: image relative path
      - metadata_fn: metadata JSON path (contains boxes, names)
      - objects: object category list
      - question: question (with [N] references)
      - answer_choices: answer options
      - answer_label: correct answer index
      - rationale_choices: rationale options
      - rationale_label: correct rationale index
    """
    jsonl_path = os.path.join(vcr_root, f"{split}.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"VCR annotation file not found: {jsonl_path}")
    
    annotations = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ann = json.loads(line)
            annotations.append(ann)
            if max_samples and len(annotations) >= max_samples * 3:
                # Load extra for later filtering
                break
    
    logger.info(f"Loaded VCR {split} annotations: {len(annotations)} entries")
    return annotations


def get_scene_hint(ann: Dict) -> str:
    """
    Extract minimal scene hint from VCR annotation (no object detection / bounding boxes).
    
    Only provides movie name metadata; GPT relies entirely on its own visual capabilities.
    """
    movie = ann.get("movie", "")
    if movie:
        return f"This is a movie still (source: {movie})."
    return "This is a movie still."


def select_task_type(ann: Dict) -> str:
    """
    Select task type based on VCR annotation metadata.
    
    Since we no longer provide object detection results to GPT, we only use the
    objects field from annotations for rough scene judgment to adjust sampling weights.
    GPT decides question content by looking at the image itself.
    """
    objects = ann.get("objects", [])
    num_persons = sum(1 for o in objects if o == "person")
    num_objects = len(objects)
    has_vehicle = any(o in ["car", "truck", "bus", "motorcycle", "bicycle"] for o in objects)
    has_furniture = any(o in ["chair", "table", "couch", "bed", "desk"] for o in objects)
    
    # Build candidate list (with weights) — use Batch 2 weights if diversity mode is on
    base_weights = TASK_WEIGHTS_BATCH2 if _use_batch2_diversity else TASK_WEIGHTS
    candidates = dict(base_weights)  # copy weights
    
    if num_persons >= 2:
        candidates["perspective_taking"] *= 1.8
        candidates["temporal_reasoning"] *= 1.5
        candidates["dynamic_simulation"] *= 1.5  # multi-person scenes suit dynamic simulation
    
    if has_vehicle:
        candidates["physical_intuition"] *= 1.5
        candidates["temporal_reasoning"] *= 1.3
        candidates["spatial_relation"] *= 1.3  # vehicle scenes suit spatial reasoning
    
    if has_furniture:
        candidates["perspective_taking"] *= 1.3
        candidates["visual_counterfactual"] *= 1.3
        candidates["spatial_relation"] *= 1.5  # indoor scenes suit spatial reasoning
    
    if num_objects >= 5:
        candidates["spatial_relation"] *= 1.5  # many objects suit spatial reasoning
        candidates["dynamic_simulation"] *= 1.3  # complex scenes suit dynamic simulation
    
    # Normalize and sample
    total = sum(candidates.values())
    probs = {k: v / total for k, v in candidates.items()}
    
    r = random.random()
    cumsum = 0.0
    for task_type, prob in probs.items():
        cumsum += prob
        if r <= cumsum:
            return task_type
    
    return "temporal_reasoning"  # fallback


# Global task type balance counter (ensures coverage of all types in small batches)
_task_type_counter = Counter()
_task_type_lock = None  # lazy init to avoid fork issues


def select_task_type_balanced(ann: Dict, num_samples: int) -> str:
    """
    Balanced task type selection: uses round-robin in small batches to ensure
    all types are covered; falls back to probability sampling for large batches.
    
    Strategy:
      - First N*2 samples (N = number of task types) use round-robin, ensuring each type appears at least twice
      - After that, use select_task_type's probability sampling
    """
    import threading
    global _task_type_lock
    if _task_type_lock is None:
        _task_type_lock = threading.Lock()
    
    all_types = list(TASK_PROMPTS.keys())
    num_types = len(all_types)
    warmup_count = num_types * 2  # first 12 samples use round-robin
    
    with _task_type_lock:
        total_assigned = sum(_task_type_counter.values())
        
        if total_assigned < warmup_count:
            # Round-robin phase: find the type with the lowest count
            min_count = min(_task_type_counter.get(t, 0) for t in all_types)
            underrepresented = [t for t in all_types if _task_type_counter.get(t, 0) == min_count]
            # Randomly pick from underrepresented types
            chosen = random.choice(underrepresented)
            _task_type_counter[chosen] += 1
            return chosen
        else:
            # Probability sampling phase: use original select_task_type with balance correction
            chosen = select_task_type(ann)
            # If a type already exceeds target ratio by 50%, resample once
            target_ratio = TASK_WEIGHTS.get(chosen, 0.15)
            actual_ratio = _task_type_counter.get(chosen, 0) / max(total_assigned, 1)
            if actual_ratio > target_ratio * 1.5:
                chosen = select_task_type(ann)  # resample
            _task_type_counter[chosen] += 1
            return chosen


def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image to base64"""
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Image encoding failed: {image_path}, {e}")
        return None


# ============================================================
# GPT API calls
# ============================================================

def create_openai_client(base_url: str = None):
    """
    Create OpenAI client (internal iChat API).
    
    Internal API does not require a real api_key; uses extra_headers for authentication.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Please install openai SDK: pip install openai>=1.0.0")
        sys.exit(1)
    
    url = base_url or ICHAT_BASE_URL
    # Disable SDK's built-in retry (max_retries=0) because it reuses stale HMAC headers.
    # We handle retries ourselves in call_gpt5_with_image() with fresh auth on each attempt.
    return OpenAI(api_key="ichat", base_url=url, max_retries=0)


def call_gpt5_with_image(
    client,
    model: str,
    system_prompt: str,
    image_base64: str,
    auth_config: Dict[str, str] = None,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Optional[str]:
    """
    Call internal iChat API: send image and prompt, get structured output.
    
    Args:
        client: OpenAI client
        model: model name (gpt-5, gpt-4o, etc.)
        system_prompt: system prompt
        image_base64: base64-encoded image
        auth_config: auth config {"source": ..., "appid": ..., "appkey": ..., "rtx": ...}
        max_retries: maximum retry count
        temperature: sampling temperature
        max_tokens: maximum output token count
    
    Returns:
        GPT text response, or None (on failure)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": "Please carefully observe this movie still. Following the system prompt requirements, construct a high-quality reasoning question and generate the complete reasoning process. Output strictly in JSON format.",
                },
            ],
        },
    ]
    
    for attempt in range(max_retries):
        wait_time = (2 ** attempt) + random.random()  # default wait time
        try:
            # Recalculate auth for each request (timestamp changes)
            extra_headers = get_auth_headers(
                source=auth_config["source"],
                appid=auth_config["appid"],
                appkey=auth_config["appkey"],
            )
            extra_body = {"cid": auth_config["rtx"]}
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=300,
                response_format={"type": "json_object"},
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                logger.warning(f"API returned empty content (attempt {attempt+1}/{max_retries})")
        except Exception as e:
            error_str = str(e)
            # Distinguish retryable vs non-retryable errors
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = (2 ** (attempt + 2)) + random.random() * 5  # wait longer for rate limit
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), "
                              f"waiting {wait_time:.1f}s before retry")
            elif "invalid_api_key" in error_str.lower() or "401" in error_str:
                # Distinguish HMAC time window expiry from real auth failure
                if "超出时间窗口" in error_str or "time" in error_str.lower():
                    # HMAC timestamp expired — retryable (will recalculate auth on next attempt)
                    wait_time = 2 + random.random()
                    logger.warning(f"HMAC time window expired (attempt {attempt+1}/{max_retries}), "
                                  f"will recalculate auth and retry in {wait_time:.1f}s")
                else:
                    logger.error(f"Auth failed, aborting retries: {e}")
                    return None  # non-retryable
            else:
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {e}, "
                              f"waiting {wait_time:.1f}s before retry")
        time.sleep(wait_time)
    
    return None


# ============================================================
# Response parsing and validation
# ============================================================

def parse_gpt_response(response_text: str) -> Optional[Dict]:
    """
    Parse GPT's JSON response.
    
    Handles common format issues:
      - markdown code block wrapping
      - trailing commas
      - incomplete JSON (truncated by max_tokens)
      - unclosed strings/arrays/objects
    """
    if not response_text:
        return None
    
    # Remove markdown code blocks
    text = response_text.strip()
    if text.startswith("```"):
        # Remove ```json and ```
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    
    # First attempt: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: remove trailing commas + truncate to last complete }
    text_fixed = re.sub(r',\s*([}\]])', r'\1', text)
    last_brace = text_fixed.rfind('}')
    if last_brace > 0:
        text_fixed = text_fixed[:last_brace + 1]
    try:
        return json.loads(text_fixed)
    except json.JSONDecodeError:
        pass
    
    # Third attempt: repair truncated JSON (caused by max_tokens truncation)
    # Strategy: try to close uncompleted strings, arrays, and objects
    try:
        repaired = _repair_truncated_json(text)
        if repaired:
            return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass
    
    logger.debug(f"JSON parse completely failed, first 200 chars: {text[:200]}")
    return None


def _repair_truncated_json(text: str) -> Optional[str]:
    """
    Attempt to repair a truncated JSON string.
    
    Common truncation scenarios:
      - String truncated mid-way: {"key": "val  -> {"key": "val"}
      - Array truncated mid-way: [1, 2, 3  -> [1, 2, 3]
      - Nested structure truncated
    """
    if not text or not text.strip().startswith('{'):
        return None
    
    # Remove trailing commas
    text = re.sub(r',\s*$', '', text.rstrip())
    
    # Count unclosed brackets
    in_string = False
    escape_next = False
    stack = []  # track unclosed bracket types
    
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
    
    # If truncated mid-string, close the string first
    if in_string:
        text += '"'
    
    # Remove potentially incomplete trailing key-value pairs
    # Then remove trailing commas
    text = re.sub(r',\s*$', '', text.rstrip())
    
    # Close all unclosed brackets (in reverse order)
    for bracket in reversed(stack):
        if bracket == '{':
            text += '}'
        elif bracket == '[':
            text += ']'
    
    return text


def validate_sample(data: Dict) -> Tuple[bool, str]:
    """
    Validate generated sample quality (two-phase approach).
    
    Checks:
      1. Required fields exist
      2. [LATENT_START] and [LATENT_END] markers exist and appear exactly once each
      3. Substantive reasoning content before and after markers
      4. Marked paragraph has sufficient visual thinking content
      5. latent_key_tokens format is correct
      6. reasoning_full is sufficiently detailed
      7. answer is concise
    """
    # 1. Required fields
    required_fields = ["question", "answer", "reasoning_full", "latent_key_tokens"]
    for field_name in required_fields:
        if field_name not in data or not data[field_name]:
            return False, f"Missing field: {field_name}"
    
    reasoning = data["reasoning_full"]
    
    # 2. [LATENT_START] and [LATENT_END] marker check
    start_count = reasoning.count("[LATENT_START]")
    end_count = reasoning.count("[LATENT_END]")
    if start_count == 0:
        return False, "Missing [LATENT_START] marker"
    if end_count == 0:
        return False, "Missing [LATENT_END] marker"
    if start_count > 1:
        return False, f"[LATENT_START] appears {start_count} times, should appear exactly once"
    if end_count > 1:
        return False, f"[LATENT_END] appears {end_count} times, should appear exactly once"
    
    start_pos = reasoning.find("[LATENT_START]")
    end_pos = reasoning.find("[LATENT_END]")
    if start_pos >= end_pos:
        return False, "[LATENT_START] must come before [LATENT_END]"
    
    # 3. Content before/after markers check
    before_latent = reasoning[:start_pos].strip()
    latent_text = reasoning[start_pos + len("[LATENT_START]"):end_pos].strip()
    after_latent = reasoning[end_pos + len("[LATENT_END]"):].strip()
    
    if len(before_latent) < 30:
        return False, f"Reasoning before [LATENT_START] too short: {len(before_latent)} chars"
    if len(after_latent) < 20:
        return False, f"Conclusion after [LATENT_END] too short: {len(after_latent)} chars"
    if len(latent_text) < 30:
        return False, f"Marked visual thinking paragraph too short: {len(latent_text)} chars"
    
    # Check for substantive reasoning before marker (contains reasoning keywords)
    reasoning_keywords = ["need", "based on", "because", "therefore", "principle", "rule",
                         "analy", "observ", "judg", "infer", "consider", "hypothe",
                         "if ", "then", "first", "second", "key", "require",
                         "notice", "see ", "look", "examin", "question", "determine",
                         "must", "should", "would", "could", "image", "scene",
                         "this", "the ", "from", "to ", "understand", "assess"]
    has_reasoning = sum(1 for kw in reasoning_keywords if kw.lower() in before_latent.lower())
    if has_reasoning < 2:
        return False, f"Insufficient reasoning keywords before marker (only {has_reasoning} hits, need at least 2)"
    
    # 4. latent_key_tokens format check (staged format)
    tokens = data["latent_key_tokens"]
    if not isinstance(tokens, list) or len(tokens) < 2:
        return False, f"latent_key_tokens needs at least 2 stages, got {len(tokens) if isinstance(tokens, list) else 0}"
    if len(tokens) > 8:
        return False, f"latent_key_tokens has {len(tokens)} stages, max 8"
    
    # 5. Check staged key_tokens quality
    language_only_words = {"because", "therefore", "however", "but", "infer", "summarize",
                          "conclude", "first", "second", "finally", "moreover", "although"}
    total_token_count = 0
    for i, stage in enumerate(tokens):
        # Validate stage structure
        if not isinstance(stage, dict):
            return False, f"latent_key_tokens[{i}] is not a dict: {type(stage)}"
        if "stage" not in stage or "tokens" not in stage:
            return False, f"latent_key_tokens[{i}] missing 'stage' or 'tokens' key"
        if not isinstance(stage["stage"], str) or not stage["stage"].strip():
            return False, f"latent_key_tokens[{i}] has empty stage name"
        if not isinstance(stage["tokens"], list) or len(stage["tokens"]) == 0:
            return False, f"Stage '{stage['stage']}' has no tokens"
        
        total_token_count += len(stage["tokens"])
        
        # Validate individual tokens within stage
        for token in stage["tokens"]:
            if not isinstance(token, str):
                return False, f"Stage '{stage['stage']}' contains non-string token: {token}"
            if token.lower() in language_only_words:
                return False, f"Stage '{stage['stage']}' contains pure language word '{token}'"
            if len(token) > 50:
                return False, f"Token '{token}' in stage '{stage['stage']}' too long ({len(token)} chars)"
    
    if total_token_count < 3:
        return False, f"Total tokens across all stages too few: {total_token_count}, need at least 3"
    if total_token_count > 30:
        return False, f"Total tokens across all stages too many: {total_token_count}, max 30"
    
    # 6. Length checks
    if len(data["question"]) < 15:
        return False, f"Question too short: {len(data['question'])} chars"
    if len(data["answer"]) < 10:
        return False, f"Answer too short: {len(data['answer'])} chars"
    if len(data["answer"]) > 500:
        return False, f"Answer too long ({len(data['answer'])} chars), answer should be a concise conclusion"
    
    # 7. reasoning_full length check
    if len(reasoning) < 150:
        return False, f"reasoning_full too short ({len(reasoning)} chars), should contain detailed reasoning"
    
    return True, "passed"


# ============================================================
# Single sample processing pipeline
# ============================================================

def process_single_sample(
    client,
    model: str,
    ann: Dict,
    vcr_root: str,
    auth_config: Dict[str, str] = None,
    task_type: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> Optional[LatentCoTSample]:
    """
    Process a single VCR sample: load image -> select task type -> call GPT -> parse & validate.
    
    Args:
        client: OpenAI client
        model: model name
        ann: VCR annotation data
        vcr_root: VCR data root directory
        auth_config: authentication config
        task_type: specified task type (None for auto-selection)
        temperature: sampling temperature
        max_tokens: maximum output token count
    
    Returns:
        LatentCoTSample or None (on failure)
    """
    # 1. Load image
    img_fn = ann.get("img_fn", "")
    image_path = os.path.join(vcr_root, "vcr1images", img_fn)
    
    if not os.path.exists(image_path):
        return None
    
    image_b64 = encode_image_base64(image_path)
    if not image_b64:
        return None
    
    # 2. Get scene hint (no object detection / bounding boxes)
    scene_hint = get_scene_hint(ann)
    
    # 3. Select task type
    if task_type is None:
        task_type = select_task_type(ann)
    
    # 4. Build prompt (only scene hint, no object/bbox info)
    prompt_template = TASK_PROMPTS[task_type]
    system_prompt = prompt_template.format(
        scene_hint=scene_hint,
        division_constraint=DIVISION_CONSTRAINT,
        key_concepts_instruction=KEY_CONCEPTS_INSTRUCTION,
    )
    
    # 4b. Append diversity enhancement for Batch 2
    if _use_batch2_diversity:
        system_prompt += BATCH2_DIVERSITY_INSTRUCTION
    
    # 5. Call GPT
    response_text = call_gpt5_with_image(
        client=client,
        model=model,
        system_prompt=system_prompt,
        image_base64=image_b64,
        auth_config=auth_config,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    if not response_text:
        return None
    
    # 6. Parse response
    logger.debug(f"API response length: {len(response_text)} chars ({img_fn})")
    parsed = parse_gpt_response(response_text)
    if not parsed:
        logger.warning(f"JSON parse failed: {img_fn} (response length: {len(response_text)}, first 100 chars: {response_text[:100]})")
        return None
    
    # 7. Validate
    is_valid, reason = validate_sample(parsed)
    if not is_valid:
        logger.debug(f"Sample validation failed ({img_fn}): {reason}")
        return None
    
    # 8. Extract latent_text from reasoning_full
    reasoning_full = parsed["reasoning_full"]
    start_pos = reasoning_full.find("[LATENT_START]")
    end_pos = reasoning_full.find("[LATENT_END]")
    latent_text = reasoning_full[start_pos + len("[LATENT_START]"):end_pos].strip()
    
    # 9. Build output
    return LatentCoTSample(
        image=os.path.join("vcr1images", img_fn),
        image_path=image_path,
        question=parsed["question"],
        answer=parsed["answer"],
        task_type=task_type,
        reasoning_full=reasoning_full,
        latent_text=latent_text,
        latent_key_tokens=parsed.get("latent_key_tokens", []),
    )


# ============================================================
# Post-processing: [LATENT] -> <|pause|> conversion
# ============================================================

def post_process_sample(sample: LatentCoTSample) -> Dict:
    """
    Post-process: convert LatentCoTSample to final training format (two-phase approach).
    
    Key conversions:
      - [LATENT_START]...[LATENT_END] in reasoning_full -> [LATENT] (reasoning_with_latent)
      - [LATENT] -> <|pause|> (reasoning_for_training)
      - Extract latent_text and latent_key_tokens
    """
    reasoning_full = sample.reasoning_full
    
    # Generate reasoning_with_latent: remove marked paragraph, replace with [LATENT]
    start_pos = reasoning_full.find("[LATENT_START]")
    end_pos = reasoning_full.find("[LATENT_END]")
    reasoning_with_latent = (
        reasoning_full[:start_pos].rstrip() + 
        " [LATENT] " + 
        reasoning_full[end_pos + len("[LATENT_END]"):].lstrip()
    )
    
    # Generate reasoning_for_training: [LATENT] -> <|pause|>
    reasoning_for_training = reasoning_with_latent.replace("[LATENT]", "<|pause|>")
    
    # Calculate [LATENT] relative position
    latent_char_pos = reasoning_with_latent.find("[LATENT]")
    total_chars = len(reasoning_with_latent)
    latent_relative_pos = latent_char_pos / max(total_chars, 1)
    
    # Serialize
    result = {
        "image": sample.image,
        "image_path": sample.image_path,
        "question": sample.question,
        "answer": sample.answer,
        "task_type": sample.task_type,
        "reasoning_full": sample.reasoning_full,  # full CoT with [LATENT_START]...[LATENT_END]
        "reasoning_with_latent": reasoning_with_latent,  # [LATENT] version
        "reasoning_for_training": reasoning_for_training,  # <|pause|> version for training
        "latent_text": sample.latent_text,  # original text that was replaced
        "latent_key_tokens": sample.latent_key_tokens,  # key tokens grouped by cognitive stages
        "latent_position": round(latent_relative_pos, 3),
        "num_stages": len(sample.latent_key_tokens),
        "num_latent_tokens": sum(len(s["tokens"]) for s in sample.latent_key_tokens if isinstance(s, dict)),
    }
    
    return result


# ============================================================
# Batch processing and concurrency control
# ============================================================

def process_batch(
    client,
    model: str,
    annotations: List[Dict],
    vcr_root: str,
    num_samples: int,
    auth_config: Dict[str, str] = None,
    workers: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    save_interval: int = 100,
    output_path: str = None,
    exclude_images: set = None,
) -> List[Dict]:
    """
    Batch process VCR samples.
    
    Args:
        client: OpenAI client
        model: model name
        annotations: VCR annotation list
        vcr_root: VCR data root directory
        num_samples: target sample count
        auth_config: authentication config
        workers: concurrent thread count
        temperature: sampling temperature
        max_tokens: maximum output token count
        save_interval: save intermediate results every N samples
        output_path: output path (for intermediate saves)
        exclude_images: set of image paths to exclude (e.g., from previous batch)
    
    Returns:
        List of processed training data
    """
    results = []
    stats = {
        "total_attempted": 0,
        "success": 0,
        "failed_image": 0,
        "failed_api": 0,
        "failed_parse": 0,
        "failed_validate": 0,
        "task_type_dist": Counter(),
    }
    
    if exclude_images is None:
        exclude_images = set()
    
    # Shuffle annotations
    random.shuffle(annotations)
    
    # Pre-load existing image files into memory set (avoids slow per-file os.path.exists on ceph)
    vcr_images_dir = os.path.join(vcr_root, "vcr1images")
    logger.info(f"Pre-scanning image directory: {vcr_images_dir}")
    existing_images = set()
    scan_start = time.time()
    for subdir in os.listdir(vcr_images_dir):
        subdir_path = os.path.join(vcr_images_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in os.listdir(subdir_path):
                # Store as relative path matching img_fn format: "subdir/fname"
                existing_images.add(os.path.join(subdir, fname))
        elif os.path.isfile(subdir_path):
            existing_images.add(subdir)
    logger.info(f"Pre-scan complete: {len(existing_images)} images found in {time.time()-scan_start:.1f}s")
    
    # Deduplicate: process each image only once, filter out missing images
    seen_images = set()
    unique_annotations = []
    skipped_missing = 0
    skipped_excluded = 0
    for ann in annotations:
        img_fn = ann.get("img_fn", "")
        if not img_fn:
            continue
        # Skip images from previous batches
        if img_fn in exclude_images:
            skipped_excluded += 1
            continue
        if img_fn not in seen_images:
            # Check against pre-loaded set (O(1) lookup, no filesystem I/O)
            if img_fn not in existing_images:
                skipped_missing += 1
                continue
            seen_images.add(img_fn)
            unique_annotations.append(ann)
    
    if skipped_missing > 0:
        logger.warning(f"Skipped {skipped_missing} annotations with missing images")
    if skipped_excluded > 0:
        logger.info(f"Excluded {skipped_excluded} annotations from previous batches ({len(exclude_images)} unique images excluded)")
    
    logger.info(f"After dedup: {len(unique_annotations)} unique images (from {len(annotations)} annotations)")
    
    # Limit processing count (prepare extra considering failure rate)
    process_count = min(len(unique_annotations), int(num_samples * 2.0))
    to_process = unique_annotations[:process_count]
    
    logger.info(f"Starting to process {process_count} images, target {num_samples} valid samples")
    logger.info(f"Concurrent threads: {workers}, model: {model}")
    
    start_time = time.time()
    
    # Reset global task type counter
    global _task_type_counter
    _task_type_counter = Counter()
    
    def _process_one(ann):
        """Single sample processing function (thread-safe)"""
        # Use balanced task type selection
        balanced_task = select_task_type_balanced(ann, num_samples)
        return process_single_sample(
            client=client,
            model=model,
            ann=ann,
            vcr_root=vcr_root,
            auth_config=auth_config,
            task_type=balanced_task,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, ann): ann for ann in to_process}
        
        for future in as_completed(futures):
            stats["total_attempted"] += 1
            
            try:
                sample = future.result(timeout=120)
                
                if sample is not None:
                    result = post_process_sample(sample)
                    results.append(result)
                    stats["success"] += 1
                    stats["task_type_dist"][sample.task_type] += 1
                else:
                    stats["failed_api"] += 1
            except TimeoutError:
                stats["failed_api"] += 1
                logger.debug(f"Processing timeout (120s)")
            except Exception as e:
                stats["failed_api"] += 1
                logger.debug(f"Processing error: {type(e).__name__}: {e}")
            
            # Progress report
            if stats["total_attempted"] % 50 == 0:
                elapsed = time.time() - start_time
                rate = stats["total_attempted"] / elapsed * 3600
                success_rate = stats["success"] / max(stats["total_attempted"], 1) * 100
                logger.info(
                    f"Progress: {stats['total_attempted']}/{process_count} "
                    f"(success {stats['success']}, rate {success_rate:.1f}%, "
                    f"speed {rate:.0f}/h)"
                )
            
            # Intermediate save
            if output_path and stats["success"] % save_interval == 0 and stats["success"] > 0:
                _save_intermediate(results, output_path, stats)
            
            # Stop early if target count reached
            if stats["success"] >= num_samples:
                logger.info(f"Reached target count {num_samples}, stopping early")
                # Cancel remaining tasks
                for f in futures:
                    f.cancel()
                break
    
    # Truncate to target count
    results = results[:num_samples]
    
    # Final statistics
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete! Elapsed {elapsed/60:.1f} minutes")
    logger.info(f"  Attempted: {stats['total_attempted']}")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Success rate: {stats['success']/max(stats['total_attempted'],1)*100:.1f}%")
    logger.info(f"  Task type distribution:")
    for task_type, count in sorted(stats["task_type_dist"].items()):
        logger.info(f"    {task_type}: {count} ({count/max(stats['success'],1)*100:.1f}%)")
    
    return results


def _save_intermediate(results: List[Dict], output_path: str, stats: Dict):
    """Save intermediate results"""
    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Intermediate save: {len(results)} samples -> {tmp_path}")
    except Exception as e:
        logger.warning(f"Intermediate save failed: {e}")


# ============================================================
# Dataset quality analysis
# ============================================================

def analyze_dataset(data: List[Dict]):
    """Analyze generated dataset quality"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset Quality Analysis")
    logger.info(f"{'='*60}")
    logger.info(f"  Total samples: {len(data)}")
    
    # Task type distribution
    task_dist = Counter(d["task_type"] for d in data)
    logger.info(f"\n  Task type distribution:")
    for task, cnt in sorted(task_dist.items()):
        logger.info(f"    {task}: {cnt} ({cnt/len(data)*100:.1f}%)")
    
    # Reasoning length statistics (using reasoning_full)
    reasoning_lens = [len(d.get("reasoning_full", d.get("reasoning_with_latent", ""))) for d in data]
    logger.info(f"\n  Reasoning length (chars):")
    logger.info(f"    Mean: {sum(reasoning_lens)/len(reasoning_lens):.0f}")
    logger.info(f"    Min: {min(reasoning_lens)}")
    logger.info(f"    Max: {max(reasoning_lens)}")
    
    # [LATENT] position analysis (prefer reasoning_with_latent, fallback to old format)
    latent_positions = []
    for d in data:
        r = d.get("reasoning_with_latent", d.get("reasoning_full", ""))
        pos = r.find("[LATENT]")
        if pos < 0:
            pos = r.find("[LATENT_START]")
        if pos >= 0:
            relative_pos = pos / max(len(r), 1)
            latent_positions.append(relative_pos)
    
    if latent_positions:
        logger.info(f"\n  [LATENT] relative position:")
        logger.info(f"    Mean: {sum(latent_positions)/len(latent_positions):.2f}")
        logger.info(f"    (0=start, 1=end, ideal ~0.4-0.7)")
    
    # latent_key_tokens statistics (staged format)
    all_tokens = []
    stage_counts = Counter()
    total_token_counts = []
    for d in data:
        stages = d.get("latent_key_tokens", [])
        sample_token_count = 0
        for stage in stages:
            if isinstance(stage, dict):
                stage_name = stage.get("stage", "unknown")
                stage_tokens = stage.get("tokens", [])
                stage_counts[stage_name] += 1
                all_tokens.extend(stage_tokens)
                sample_token_count += len(stage_tokens)
            else:
                # Backward compatibility with flat list format
                all_tokens.append(stage)
                sample_token_count += 1
        total_token_counts.append(sample_token_count)
    
    token_freq = Counter(all_tokens)
    logger.info(f"\n  latent_key_tokens statistics (staged):")
    logger.info(f"    Total tokens: {len(all_tokens)}")
    logger.info(f"    Unique tokens: {len(token_freq)}")
    logger.info(f"    Top-20 frequent tokens:")
    for token, cnt in token_freq.most_common(20):
        logger.info(f"      {token}: {cnt}")
    
    # Stage count distribution
    num_stages_dist = Counter()
    for d in data:
        stages = d.get("latent_key_tokens", [])
        num_stages_dist[len(stages)] += 1
    
    logger.info(f"\n  Cognitive stage count distribution:")
    for num_stages, cnt in sorted(num_stages_dist.items()):
        logger.info(f"    {num_stages} stages: {cnt} ({cnt/len(data)*100:.1f}%)")
    
    # Total token count distribution
    token_count_dist = Counter(total_token_counts)
    logger.info(f"\n  Total tokens per sample distribution:")
    for num_tokens, cnt in sorted(token_count_dist.items()):
        logger.info(f"    {num_tokens} tokens: {cnt} ({cnt/len(data)*100:.1f}%)")
    
    # Top stage names
    logger.info(f"\n  Top-20 stage names:")
    for stage_name, cnt in stage_counts.most_common(20):
        logger.info(f"    {stage_name}: {cnt}")


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Latent Reasoning QA+CoT data for VCR images via internal iChat API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (set APPID/APPKEY via environment variables)
  export ICHAT_APPID="your_appid"
  export ICHAT_APPKEY="your_appkey"
  python scripts/generate_vcr_latent_cot.py \\
    --vcr_root data/nld_phase1/raw/vcr \\
    --output data/nld_phase1/vcr_latent_cot.json \\
    --rtx your_rtx \\
    --num_samples 5000

  # Pass auth info via command-line arguments
  python scripts/generate_vcr_latent_cot.py \\
    --vcr_root data/nld_phase1/raw/vcr \\
    --output data/nld_phase1/vcr_latent_cot.json \\
    --rtx your_rtx \\
    --appid your_appid \\
    --appkey your_appkey \\
    --model gpt-4o \\
    --num_samples 10000 \\
    --workers 16

  # Debug mode (small sample count)
  python scripts/generate_vcr_latent_cot.py \\
    --vcr_root data/nld_phase1/raw/vcr \\
    --output data/nld_phase1/vcr_latent_cot_debug.json \\
    --rtx your_rtx \\
    --num_samples 10 \\
    --workers 2 \\
    --verbose
        """,
    )
    
    # Path arguments
    parser.add_argument("--vcr_root", type=str, required=True,
                        help="VCR data root directory (containing train.jsonl and vcr1images/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    
    # Internal API auth arguments
    parser.add_argument("--rtx", type=str, required=True,
                        help="RTX account (for API auth and cid)")
    parser.add_argument("--appid", type=str, default=None,
                        help="Internal API AppID (or set via ICHAT_APPID env var)")
    parser.add_argument("--appkey", type=str, default=None,
                        help="Internal API AppKey (or set via ICHAT_APPKEY env var)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Custom API endpoint URL (default: http://ichat.woa.com/api/external)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (default: gpt-4o)")
    
    # Data arguments
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Target sample count (default: 5000)")
    parser.add_argument("--split", type=str, default="train",
                        help="VCR dataset split (default: train)")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=list(TASK_PROMPTS.keys()),
                        help="Specify task type (default: auto-select)")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum output token count (default: 8192)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent thread count (default: 8)")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Intermediate save interval (default: every 100 samples)")
    
    # Batch 2 diversity options
    parser.add_argument("--exclude_images", type=str, default=None,
                        help="Path to previous batch JSON file; images in it will be excluded")
    parser.add_argument("--batch2_diversity", action="store_true",
                        help="Enable Batch 2 diversity mode: adjusted task weights + diversity-enhanced prompts")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging output")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    random.seed(args.seed)
    
    # Enable Batch 2 diversity mode if requested
    global _use_batch2_diversity
    if args.batch2_diversity:
        _use_batch2_diversity = True
        logger.info("Batch 2 diversity mode ENABLED: adjusted task weights + diversity-enhanced prompts")
        logger.info(f"  Task weights: {TASK_WEIGHTS_BATCH2}")
    
    # Load exclude images from previous batch
    exclude_images_set = set()
    if args.exclude_images:
        logger.info(f"Loading exclude images from: {args.exclude_images}")
        try:
            with open(args.exclude_images, 'r') as f:
                prev_data = json.load(f)
            for item in prev_data:
                img = item.get('image', '')
                # Extract img_fn (e.g., "movieX/img.jpg" from "vcr1images/movieX/img.jpg")
                if img.startswith('vcr1images/'):
                    img_fn = img[len('vcr1images/'):]
                else:
                    img_fn = img
                if img_fn:
                    exclude_images_set.add(img_fn)
            logger.info(f"  Loaded {len(exclude_images_set)} images to exclude")
        except Exception as e:
            logger.error(f"Failed to load exclude images: {e}")
            sys.exit(1)
    
    # Get internal API auth info
    appid = args.appid or os.environ.get("ICHAT_APPID")
    appkey = args.appkey or os.environ.get("ICHAT_APPKEY")
    if not appid or not appkey:
        logger.error("Please provide auth info via --appid/--appkey args or ICHAT_APPID/ICHAT_APPKEY env vars")
        sys.exit(1)
    
    auth_config = {
        "source": args.rtx,
        "appid": appid,
        "appkey": appkey,
        "rtx": args.rtx,
    }
    
    # Validate paths
    if not os.path.isdir(args.vcr_root):
        logger.error(f"VCR root directory not found: {args.vcr_root}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 1. Load VCR annotations
    # When excluding images from previous batches, load more annotations to compensate
    effective_num_samples = args.num_samples
    if exclude_images_set:
        # Load enough to have sufficient images after exclusion
        # VCR train has ~212K annotations; load generously
        effective_num_samples = args.num_samples + len(exclude_images_set)
        logger.info(f"Adjusting load count: {args.num_samples} + {len(exclude_images_set)} excluded = {effective_num_samples}")
    logger.info(f"Loading VCR annotations: {args.vcr_root}/{args.split}.jsonl")
    annotations = load_vcr_annotations(args.vcr_root, args.split, effective_num_samples)
    
    # 2. Create OpenAI client (internal iChat API)
    logger.info(f"Creating iChat API client (model: {args.model}, RTX: {args.rtx})")
    client = create_openai_client(args.base_url)
    
    # 3. Batch processing
    logger.info(f"Starting Latent Reasoning data generation...")
    results = process_batch(
        client=client,
        model=args.model,
        annotations=annotations,
        vcr_root=args.vcr_root,
        num_samples=args.num_samples,
        auth_config=auth_config,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_interval=args.save_interval,
        output_path=args.output,
        exclude_images=exclude_images_set,
    )
    
    # 4. Dataset quality analysis
    if results:
        analyze_dataset(results)
    
    # 5. Save final results
    logger.info(f"\nSaving final results: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    file_size_mb = os.path.getsize(args.output) / 1024 / 1024
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info(f"  Samples: {len(results)}")
    
    # 6. Save run configuration
    config_path = args.output.replace(".json", "_config.json")
    config = {
        "model": args.model,
        "num_samples": len(results),
        "target_samples": args.num_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "vcr_root": args.vcr_root,
        "split": args.split,
        "seed": args.seed,
        "task_type": args.task_type,
        "rtx": args.rtx,
        "base_url": args.base_url or ICHAT_BASE_URL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "batch2_diversity": args.batch2_diversity,
        "exclude_images": args.exclude_images,
        "num_excluded_images": len(exclude_images_set),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"  Config saved: {config_path}")
    
    logger.info(f"\nDone! Generated {len(results)} Latent Reasoning training samples")
    
    # 7. Print next steps
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Check data quality: python -c \"import json; d=json.load(open('{args.output}')); print(d[0])\"")
    logger.info(f"  2. [LATENT] -> <|pause|> conversion is already done in reasoning_for_training field")
    logger.info(f"  3. Update training config to use this dataset")


if __name__ == "__main__":
    main()
