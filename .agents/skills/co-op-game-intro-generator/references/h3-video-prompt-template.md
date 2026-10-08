# H3 Video Prompt Template

## Portable adaptation

This is a creative planning template. User requirements override its default
15-second timeline, body proportions, equipment, environment, ratio, and palette.
Retiming must preserve readable event order. Use only supplied reference assets:
omit `{height_ref}` or identity-image bindings when absent and replace their role
with text design constraints. Compile the filled plan into the H3 base or Ref2VA
format before rendering; preserve exact visible UI strings in the user's language.

Use available tools and the actual backend's parameter schema. Audio directions
describe desired content rather than asserting that a particular API supports it.

## Template completion

Resolve the palette variables `{background_color}`, `{ui_color}`, `{text_color}`,
`{functional_accent_color}`, and `{power_accent_color}` from the selected style.
Keep the five values distinct enough for readable UI contrast. Use the power
accent for danger, exit, and warning states unless the user supplies a separate
semantic-color requirement.

Fill visible copy in the user's requested language. The following English values
are defaults only, not mandatory output:

- `{player1_label}` / `{player2_label}`: `PLAYER 1` / `PLAYER 2`
- `{ready_label}`: `READY`
- `{start_label}`, `{continue_label}`, `{settings_label}`, `{exit_label}`:
  `START NEW GAME`, `CONTINUE`, `SETTINGS`, `EXIT GAME`
- `{player1_equipment_label}`: `RIGHT ARM EQUIPMENT`
- `{player1_option_a}` / `{player1_option_b}`: `PHANTOM GRIP` / `CHRONOS CLAW`
- `{player2_equipment_label}`: `ARMAMENT CUSTOMIZATION`
- `{hand_label}`, `{forearm_label}`, `{elbow_label}`, `{upper_arm_label}`:
  `HAND`, `FOREARM`, `ELBOW`, `UPPER ARM`
- `{confirm_label}` / `{loading_label}`: `CONFIRM CONFIG` / `LOADING`

Before compiling the rendering prompt, search the filled template for every
remaining token enclosed in braces. Replace required values. For an absent optional
reference, remove its binding and rewrite every dependent clause as a text-only
constraint. Do not dispatch a prompt while any template token remains.

## Prompt principle
Use the same method as the GPT confirmation-image prompt:

**Fixed video event framework + dynamic user-style fill + locked character identity + palette-linked UI system.**

The video prompt must not blindly preserve the source prompt's default style. It must preserve the timeline, UI events, player positions, equipment logic, and text structure, while dynamically rewriting visual treatment, palette, character rendering, lighting, UI surface, icons, and city-world style according to the user's selected style.

## Priority order
1. User-selected style: {visual_style}
2. Confirmed image / UI reference: {ui_ref}
3. PLAYER 1 identity reference: {player1_ref}
4. PLAYER 2 identity reference: {player2_ref}
5. Height/body comparison reference: {height_ref}
6. Fixed source video event framework

## Reference roles
- {ui_ref}: confirmed first image. Use it to lock UI layout, menu hierarchy, color system, typography scale, button structure, character-game integration, and overall composition logic.
- {player1_ref}: PLAYER 1 identity anchor. Lock exact face, hairstyle, glasses if present, facial proportions, body identity, and nickname mapping to {player1_name}.
- {player2_ref}: PLAYER 2 identity anchor. Lock exact face, hairstyle, facial proportions, body identity, and nickname mapping to {player2_name}.
- {height_ref}: body comparison anchor. Lock the visible contrast between the two players and prevent identical body proportions.

## Global style baseline
The overall art direction must prioritize the user's selected {visual_style}.
Preserve these fixed qualities: a game main-menu interface, premium game-trailer
finish, deep integration between the UI and characters, modern commercial game
UI design, strong visual impact, clean composition, and restrained decoration.
Derive the remaining visual treatment from {visual_style}: character rendering,
expressions, costume language, color system, lighting temperature, UI materials,
button icons, typography texture, and the world revealed after loading.

## Palette system
Derive the complete video palette from {visual_style}, with linked UI colors:

- Use {background_color} as the primary background and world color.
- Use {ui_color} as the main UI color.
- Use {text_color} as the text color.
- Use {functional_accent_color} as PLAYER 1's functional accent color.
- Use {power_accent_color} as PLAYER 2's power accent and for danger, exit,
  and warning states.
- Limit the complete palette to five colors.
- Use a vivid, modern, high-contrast color language consistent with
  {visual_style}.

Menus, equipment panels, player cards, buttons, the HUD, loading bars, icons,
and text must share this color system. Do not introduce unrelated colors.

## Character identity and style lock
PLAYER 1 follows the facial identity in {player1_ref}. Preserve the face,
facial proportions, hairstyle, glasses when present, personal identity, and the
mapping to the nickname {player1_name}. Adapt expression, rendering, clothing,
and mechanical equipment to {visual_style}. PLAYER 1 always stays on the left,
with a taller, slender, agile build. Equipment uses the functional accent color,
and the mechanical claw is light, narrow, and flexible.

PLAYER 2 follows the facial identity in {player2_ref}. Preserve the face,
facial proportions, hairstyle, personal identity, and the mapping to the
nickname {player2_name}. Adapt expression, rendering, clothing, and mechanical
equipment to {visual_style}. PLAYER 2 always stays on the right, with a shorter,
broader, power-oriented build. Equipment uses {power_accent_color}, and the
mechanical fist is broad, heavy, and visibly weighty.

Do not swap character identities, merge faces, swap nicknames, or converge the
two body types.

## Fixed timeline framework

### [0-2 seconds] - Two-player main menu

Composition and camera: A high-angle extreme wide shot follows the composition
logic of {ui_ref}. The camera tilts slightly downward and slowly pushes in.

Visuals: PLAYER 1 ({player1_name}) and PLAYER 2 ({player2_name}) sit side by
side at the center. PLAYER 1 remains on the left and PLAYER 2 on the right. They
look up toward the camera with only natural breathing, blinking, and subtle body
movement.

UI: The upper-left player card displays `{player1_label}`, `{player1_name}`, and
`{ready_label}` exactly as filled. The second card in the upper-center or
upper-left two-card system displays `{player2_label}`, `{player2_name}`, and
`{ready_label}` exactly as filled. The vertical menu on the right displays
`{start_label}`, `{continue_label}`, `{settings_label}`, and `{exit_label}`
exactly as filled. `{continue_label}` is the visual center and primary
highlighted button.

Dynamic style fill: Adapt the menu background, ground texture, button shapes,
icons, typography, borders, glow, and sticker treatment to {visual_style}, while
preserving the layout and hierarchy in {ui_ref}.

Sound: Menu ambience, subtle UI-hover sounds, and a low electronic atmosphere
before the click. Ignore this direction when generating silent video.

### [2-4 seconds] - PLAYER 1 right-arm configuration

Composition and camera: In a medium shot, the camera pushes smoothly from the
main menu toward PLAYER 1's right arm. PLAYER 2 remains visible and stable in the
background.

UI: The right-side menu contracts and slides away. A panel with functional-accent
guide lines slides in from the left and displays `{player1_label}` and
`{player1_equipment_label}` exactly as filled. The equipment list first
highlights `{player1_option_a}`, then moves the selection to
`{player1_option_b}`.

Action: PLAYER 1's right cuff opens automatically and a lightweight mechanism
unfolds beneath the forearm. The fingers spread as long claw-like joints slide
into position and lock one by one, briefly revealing fine wiring, miniature
pistons, and metal connectors. Functional-accent LEDs illuminate sequentially
when configuration completes.

Dynamic style fill: Adapt the mechanism, panel material, icons, lines, locking
animation, and light effects to {visual_style}. Keep the result light, precise,
and flexible, proportioned to PLAYER 1 without changing the face, hairstyle, or
main clothing design.

Sound: Precise mechanical unfolding, light UI switching tones, and small locking
clicks.

### [4-7 seconds] - PLAYER 2 heavy-arm configuration

Composition and camera: A medium shot tracks smoothly between the players and
arcs toward PLAYER 2's left side. PLAYER 1 remains in the background, quietly
examining the configured mechanical hand.

UI: A {power_accent_color} panel slides in and displays `{player2_label}` and
`{player2_equipment_label}` exactly as filled. Its grid contains `{hand_label}`,
`{forearm_label}`, `{elbow_label}`, and `{upper_arm_label}`, with the selection
moving quickly but legibly through all four.

Action: PLAYER 2's left sleeve opens in sections. Heavy forearm plates spring
outward, the previous components detach, and new armor slides into place along
guide rails. A thick mechanical bearing replaces the elbow joint, and the broad
mechanical hand reassembles and locks. Thick wiring, hydraulic pistons, and a
dark metal skeleton appear briefly during the replacement. A restrained warm
indicator illuminates as each component locks.

Dynamic style fill: Adapt the heavy arm, UI panel, component icons, materials,
and light effects to {visual_style}. Keep it broad, weighty, and powerful, in
clear contrast with PLAYER 1's lightweight claw.

Sound: Low motor movement, heavy mechanical engagement, and weighty locking
feedback.

### [7-8.5 seconds] - Shared configuration confirmation

Composition and camera: Pull back smoothly to a medium two-player composition,
with PLAYER 1 on the left and PLAYER 2 on the right.

UI: The two equipment panels converge at center to form a shared button that
displays `{confirm_label}` exactly as filled. Adapt its border, glow, icon, and
sticker treatment to {visual_style}, while keeping the hierarchy and text legible.

Action: The cursor clicks the button. A functional-accent energy pulse travels
through PLAYER 1's mechanical claw, while a {power_accent_color} pulse travels
through PLAYER 2's fist. All UI panels contract inward and disappear. Both
players uncross their legs and adjust their posture: PLAYER 1 lifts one knee
lightly and articulates the claw fingers in sequence; PLAYER 2 plants one foot
firmly and slowly closes the heavy fist.

Sound: Confirmation tone, two-color energy pulses, and UI contraction.

### [8.5-10 seconds] - Shared world loading

Composition and camera: A wide shot reveals a shared loading bar at the bottom.

UI: The loading bar displays `{loading_label}` exactly as filled and fills
rapidly from 0% to 100%. Its left half uses {functional_accent_color} and its
right half uses {power_accent_color}. Adapt the HUD and loading-bar shape,
border, texture, and typography to {visual_style}, while keeping them legible.

Environment transformation: Continuously transform the confirmed
{visual_style} menu background into a game world in the same style. Menu strips,
color blocks, and patterns become corresponding roads, structures, and world
elements. Preserve the confirmed palette rather than forcing a yellow-and-black
or cyberpunk city.

Key constraint: The transformation remains continuous and natural, with no hard
cut, smoke occlusion, or identity change.

Sound: Rising loading tone and an ambience transition from menu to game world.

### [10-15 seconds] - Two-player world entry

Composition and camera: Move from an extreme wide shot into a third-person
tracking view. At 100% loading, both characters stand simultaneously. The camera
descends smoothly and arcs behind them into a stable two-player cooperative view.

World style: Generate the complete world from {visual_style}. Preserve the game
opening structure: dense buildings, roads, signal lights, signs, crowds or other
moving background elements, passing vehicles, overhead wires, industrial pipes,
and a distant skyline. Materials, architecture, signage, lighting, and motion
must follow {visual_style}. Do not let the source template's cyberpunk default
override the user unless the selected style is cyberpunk.

Character relationship: Clearly show the players from behind and preserve their
body contrast. PLAYER 1 is on the left, tall and slender, with the lightweight
claw hanging naturally. PLAYER 2 is on the right, shorter and broader, with the
heavy fist slightly raised. PLAYER 1 steps forward first, PLAYER 2 follows, and
they enter the street side by side.

HUD: The HUD fades in. A mini-map appears in the upper right. Separate status
bars labeled `{player1_name}` and `{player2_name}` appear in the lower left.
The first uses {functional_accent_color}; the second uses {power_accent_color}.
A shared objective marker appears in the street ahead.

Sound: City ambience, distant vehicles, footsteps, and a HUD fade-in cue.

## Negative constraints
No extra player, duplicated character, identity or username swapping, merged bodies, unintended face/hair changes, missing player, unrequested split screen or cuts, random camera shake, floating body parts, gruesome dismemberment, unwanted weapons, style-incompatible neon, unreadable UI, random letters, misspelled usernames, extra menu options, copied game logo, or watermark. Preserve the user's chosen character designs and palette.
