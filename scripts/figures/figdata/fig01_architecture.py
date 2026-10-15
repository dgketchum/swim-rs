"""Figure 1 architecture: the display contract as data, and the visible-string rules.

Everything here is declarative: the constants that fix the composition, the
caption-ownership patterns that no visible string may match, and
``architecture_spec``, which returns the frozen ``fig01_architecture.json`` payload.
The audits that hold the spec to these rules run in ``fig01_scope.build_fig01``.
"""

from __future__ import annotations

import re
from pathlib import Path

from .cohorts import EXPECTED, FIG01_CLASS_COUNTS
from .fig04_reliability import E1_MEMBERS
from .manifest import BuildError
from .paths import _CFG5

# Strings that belong to the caption or Methods and must never reach a string
# classified title / direct_label / annotation.  Derived by hand from
# fig01_production_handoff.md section 5 ("Caption- or manuscript-owned") so the
# audit is explicit rather than an accidental substring match: the configured
# counts 60 / 66 / 50 ARE allowed in the artwork and are deliberately absent.
CAPTION_ONLY_VISIBLE_PATTERNS = [
    r"Φ",  # objective symbol
    r"σ",  # sigma
    r"\bsigma",
    r"realization",
    r"iteration",
    r"\bSSEBop\b",
    r"\bPT-?JPL\b",
    r"\bSIMS\b",
    r"\bgeeSEBAL\b",
    r"\beeMETRIC\b",
    r"\bDisALEXI\b",
    r"\bECOSTRESS\b",
    r"\bVolk\b",
    r"\bGridMET\b",
    r"\bSNODAS\b",
    r"\b45\b",
    r"\b29\b",
    r"\b63\b",
    r"\b56\b",
    r"\b31\b",
    rf"\b{EXPECTED['E3_field_years']}\b",
    r"\b200\b",
    r"field[- ]year",
    r"\bpaired\b",
    r"withheld",
    r"\bweight",
]

# Exactly the strings a *current* handoff section authorizes as visible copy even
# though a caption-owned pattern matches them.  Matching is by whole-string
# equality, never substring, so an exemption cannot widen to a longer phrase.
CAPTION_PATTERN_EXEMPTIONS: dict[str, str] = {}
# Architecture 3.1.1 (user review, 2026-08-25) retired the 'Σ year' operation in
# favour of the spelled-out 'Annual Total', so the sigma exemption that 3.1.0
# needed is gone and the r'σ' objective-notation pattern is back at full
# strength: every sigma-bearing string, upper or lower case, now fails the
# caption guard.
#
# Architecture 3.2.0 (handoff section 5.3, revised 2026-08-25) then removed
# 'spread-weighted' from the artwork -- "Put `PEST++ IES` and spread-dependent
# weighting in the caption by default. Do not retain `spread-weighted` in the
# artwork merely to fill the inverse region." -- so the last exemption is gone
# and r'\bweight' is at full strength too.  The exemption table is now empty and
# must stay empty: do not add an entry without a handoff section that authorizes
# the exact string.  Section 5.3 does allow 'spread-weighted' to RETURN on the
# incoming target route if a final-size proof shows it materially clarifies the
# member-spread encoding; that return would need this exemption restored and is
# recorded as a conditional in the architecture, not as a visible string.

VISIBLE_STRING_CLASSES = {"title", "direct_label", "annotation", "proof_only"}

# Nodes on the held-out side of the firewall, and nodes that participate in
# parameter fitting or transfer-vector construction.  No edge may run from the
# first set into the second.  Revised for architecture 3.0.0 (handoff sections
# 5.4 and 6.3): the daily balance and both class-specific parameter tokens are
# now protected alongside inverse estimation and the transfer destinations.
#
# Architecture 3.2.0 (handoff sections 5.3, 10.3 and 11) replaces the single
# 'inverse_estimation' node with the three-stage closed cycle, so every stage of
# that cycle joins the protected set: "Held-out flux and meter observations
# remain forbidden as cycle inputs" (section 10.3).  The retired node id is kept
# in the set so an accidental revival is still caught.
EVALUATION_NODES = {"flux_et", "meters"}
FITTING_NODES = {
    "inverse_estimation",
    "run_balance",
    "compare",
    "update_parameters",
    "daily_balance",
    "e1_map",
    "irrigated_params",
    "rainfed_params",
    "e2_map",
    "e3_map",
    "e0_tag",
}

# Architecture 3.2.0: the literal directed cycle required by handoff sections
# 4, 5.3, 10.3, 11 and 13.  Arrowheads must close it in this order -- the
# balance produces simulated calibration quantities, comparison with targets
# produces mismatches, and the parameter update feeds the next balance run.
FIG01_CYCLE_STAGES = ("run_balance", "compare", "update_parameters")
FIG01_CYCLE_EDGES = (
    ("run_balance", "compare"),
    ("compare", "update_parameters"),
    ("update_parameters", "run_balance"),
)
# The exit that leaves the update stage after convergence and feeds the
# displayed daily state and ET (handoff section 5.3).
FIG01_CYCLE_EXIT_EDGE = ("update_parameters", "daily_balance")
FIG01_CYCLE_EXIT_LABEL = "Conditioned Parameters"
# The acquisition-date targets enter the comparison stage, never the balance.
FIG01_CONSTRAINT_INTO = "compare"
# NDVI and daily forcing drive the forward balance.
FIG01_DRIVER_INTO = "run_balance"

# Panel (b) transfer topology, asserted before the architecture is written.
FIG01_TRANSFER_SOURCE = "e1_map"
FIG01_PARAM_TOKENS = ("irrigated_params", "rainfed_params")
FIG01_TRANSFER_DESTINATIONS = {
    "e2_map": ("irrigated_params", "rainfed_params"),
    "e3_map": ("irrigated_params",),
}
# The E3 transfer path must visibly branch here, before the E2 frame, so it can
# never be read as E2 -> E3 (handoff sections 6.1, 6.3 and 11).
FIG01_TRANSFER_BRANCH_AT = "irrigated_params"

# Architecture 3.1.0 (handoff sections 5.2, 5.4, 10.3 and 11).  The E1 example's
# simulated irrigation series is display evidence for the example record ONLY.  E3
# evaluates applied water simulated for E3 fields, so no drawn relationship --
# directed edge or neutral comparison tie -- may connect the E1 nodes below to
# the E3 meter key, directly or through any intermediate node.
FIG01_E1_APPLIED_WATER_NODES = {"applied_water"}
FIG01_E3_METER_NODE = "meters"

# The two ET traces the reader is invited to compare must be drawn against one
# date mapping at comparable horizontal extent (handoff sections 5.2, 5.4, 10.3
# and 11).  Neither may be squeezed into a side strip.
FIG01_COMMON_DATE_MAPPING_COLUMNS = ("swim_ET", "flux_ET")

# Required visible record identification (handoff sections 5.1, 11 and 13): the
# reader must not have to infer the site and year from the caption.  Architecture
# 3.2.0 set the parenthesized form and near-black colour required by handoff
# section 5.1 (then "US-Bi1 (2017)"); the 2026-08-27 directed-site override
# (see the example-source generator below) moves the record to S2 while keeping
# the treatment rules -- parenthesized form, never muted gray, no middle dot.
FIG01_RECORD_ID = "S2 (2018)"

# Handoff section 9: near-black for every reader-facing identification, title,
# row name, unit, map count line and inverse-stage label.  Muted gray is reserved
# for axes, reference rules and quiet geographic context.  Section 11 stops the
# build if the record identification is rendered as muted gray secondary text.
# 2026-08-27: the figure style guide moved all reader-facing text to full black,
# which strictly satisfies the never-muted-gray intent; the contract colour
# follows (was #202124 near-black under the r4 architecture).
FIG01_IDENTIFICATION_COLOR = "#000000"

# Handoff sections 4, 5.2 and 11: panel (a) carries FIVE quantitative plotting
# regions, not seven independent sparklines.  Root-zone depletion and irrigation
# share one coordinated region; daily SWIM-RS ET and held-out flux ET share
# another.
FIG01_MAX_PLOTTING_REGIONS = 5

# Recorded display limits for the five regions (handoff section 5.2: "The proof
# script must record the final limits and assert that no plotted value is
# clipped").  These are the CONTRACT domains the section 11 clipping check runs
# against; the proof may refine ticks and padding as a Level 1 render change so
# long as no plotted value falls outside what is recorded here.
#
# Section 5.2's candidate ranges are ETf 0.8-1.6, NDVI 0.2-1.0, forcing 0 to a
# rounded maximum, depletion and irrigation 0 to their respective rounded
# maxima, and ET 0-10 mm d-1.  The frozen S2 (2018) record is adopted for every
# one of them EXCEPT the ETf bounds: retrieval members span 0.318-1.665 at this
# site, so a 0.8 floor would clip low member marks and the candidate 1.6
# ceiling would clip the highest -- and both would hide exactly the
# disagreement the region exists to show.  The floor is therefore 0.0 and the
# ceiling 1.8.  The forcing maximum rounds S2's window maximum (ETo 8.1,
# precipitation 7.6 mm d-1) to 10 (was 16 under the US-Bi1 record).
FIG01_DISPLAY_DOMAINS = {
    "etf_ensemble": (0.0, 1.8),
    "ndvi_captures": (0.2, 1.0),
    "daily_forcing": (0.0, 10.0),
    "state_and_irrigation": (0.0, 14.0),
    "state_and_irrigation_secondary": (0.0, 22.0),
    "et_comparison": (0.0, 10.0),
}

# ---- Figure 1 example record (handoff sections 5.1, 10.2, 11) --------------
# The frozen selection.  The build compares the reconstructed record against
# these values and stops if the audited site or window has moved.  Site S2 is
# the 2026-08-27 user-directed override of the original median-rank choice
# (US-Bi1); the window is S2's earliest qualifying window under the unchanged
# step-2 rule and step-5 tiebreak.
FIG01_EXAMPLE_SITE = "S2"
# This relocated January delivery is used only to reproduce the already-frozen
# Figure 1 example-window selection. Its benchmark columns are prohibited from
# the Figure 1 display table and from every Figure 3 evaluation artifact.
FIG01_LEGACY_OPENET_DAILY = (
    Path(_CFG5.data_dir) / "_deprecated_january_delivery" / "openet_flux" / "daily_data"
)
FIG01_EXAMPLE_START = "2018-04-01"
FIG01_EXAMPLE_END = "2018-07-29"
FIG01_EXAMPLE_DAYS = 120
FIG01_EXAMPLE_CAPTURES = 8


# Column-name patterns that may never reach the Figure 1 example display table
# (handoff sections 8, 10.2 and 11).  Figure 3 owns the benchmark comparison,
# Figures 2-6 own every performance metric, and the daily filled NDVI trajectory
# is not a Figure 1 display quantity.
FIG01_FORBIDDEN_EXAMPLE_COLUMNS = [
    (r"benchmark", "OpenET benchmark series and flags are owned by Figure 3"),
    (r"interpolat", "benchmark-interpolation classification is owned by Figure 3"),
    (r"is_direct", "direct-versus-interpolated benchmark flag is owned by Figure 3"),
    (
        r"ndvi_(kcb|fill|filled|daily|interp|smooth|model)",
        "a daily filled NDVI trajectory is not a Figure 1 display quantity "
        "(handoff sections 5.2 and 11)",
    ),
    (
        r"(^|_)(kge|nse|rmse|mbe|mae|bias|r2|d_kge|rho)($|_)",
        "performance metrics belong to Figures 2-6 and the text",
    ),
    (r"(error|resid)", "evaluation residuals must not reach Figure 1"),
    (r"weight", "observation weights are caption/Methods-owned"),
]


def _assert_no_caption_facts_visible(classification: dict[str, str]) -> None:
    visible = [s for s, c in classification.items() if c in {"title", "direct_label", "annotation"}]
    hits = []
    for s in visible:
        if s in CAPTION_PATTERN_EXEMPTIONS:
            continue
        for pat in CAPTION_ONLY_VISIBLE_PATTERNS:
            if re.search(pat, s, flags=re.IGNORECASE):
                hits.append((s, pat))
    if hits:
        raise BuildError(
            "fig01: caption/Methods-owned content reached a visible string: "
            + "; ".join(f"{s!r} matches {p!r}" for s, p in hits)
        )


def architecture_spec(etf_cols: list[str], region_ranges: dict, y_axis) -> dict:
    """The Figure 1 architecture contract.

    ``etf_cols`` are the example-record ETf columns, ``region_ranges`` the observed
    data range behind each plotting region, and ``y_axis`` the region axis-block
    constructor; all three are derived from the frozen example in ``build_fig01``.
    """
    return {
        "schema_version": "3.3.0",
        "revision_3_3_0": {
            "date": "2026-08-27",
            "kind": "directed example-record change on the 3.2.0 composition",
            "changes": [
                f"the example record moves US-Bi1 (2017) -> {FIG01_RECORD_ID} under the "
                "user-directed agronomic-representativeness override recorded in "
                "fig03_example_selection.json (site from the frozen eligible list; unchanged "
                "step-2 window rule and earliest-qualifying tiebreak)",
                "recorded display domains follow the new record: etf_ensemble ceiling "
                "1.6 -> 1.8 (members span 0.318-1.665), daily_forcing maximum 16 -> 10",
                "identification colour #202124 -> #000000 (the 2026-08-27 figure style guide "
                "moved all reader-facing text to full black, which strictly satisfies the "
                "never-muted-gray intent)",
                "calibration captures in the frozen window: 15 -> 8",
            ],
            "unchanged": (
                "the 3.2.0 composition: panels, plotting regions, string classes, edges, "
                "forbidden treatments, transfer topology and every audit rule"
            ),
            "removed_strings": ["US-Bi1 (2017)"],
        },
        "supersedes_3_2_0_note": (
            "3.2.0 froze the same composition on the superseded US-Bi1 (2017) record; its "
            "review provenance below is retained verbatim"
        ),
        "supersedes": {
            "schema_version": "3.1.1",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the r3 proof's panel (a): seven stacked full-height lanes (five numbered "
                "evidence rows, the last carrying three lanes), a 'Daily State' row separate "
                "from the irrigation stems, a 'Daily Outputs' row in which the SWIM-RS and flux "
                "traces sat in separate lanes, a one-way 'Inverse Estimation' element whose "
                "iteration was implied by a single parameter-update arrow, a 'Held-Out "
                "Evaluation' label, a 'spread-weighted' label in the inverse region, and the "
                "record identified as a middle-dot 'US-Bi1 · 2017' subtitle."
            ),
            "reason": (
                "fig01_production_handoff.md was rewritten 2026-08-25 after review of revision "
                "3: 'Revision 3 reviewed; panel (a) redesign and a new Gate A proof are "
                "required'. Section 16 declares it a Level 2 contract change because it changes "
                "visible copy, panel grouping, and inverse-estimation relationships. Panel (b) "
                "is explicitly NOT superseded: section 6.1 accepts its revision-3 core topology."
            ),
            "removed_requirements": [
                "the seven-lane panel (a) structure -- five numbered evidence rows with three "
                "lanes on the last; section 4 now requires five quantitative plotting regions "
                "and section 11 stops the build above five",
                "the separate 'Daily State' region; root-zone depletion and the irrigation stems "
                "are now one coordinated region with explicit scales for both (section 5.2 "
                "item 4)",
                "the separate 'Daily Outputs' region with daily_et and flux_et in distinct "
                "lanes; the two ET traces now share one axis, one date support and one vertical "
                "scale (section 5.2 item 5)",
                "the one-way inverse representation -- a titled element plus a single "
                "parameter-update arrow; section 5.3 requires a literal closed cycle and forbids "
                "'two disconnected L-shaped legs or a generic one-way transformation'",
                "the 'Inverse Estimation' title; sections 4, 5.3 and 13 name only the three "
                "cycle verbs and section 9 speaks of 'inverse-stage labels'",
                "the 'Held-Out Evaluation' label; section 5.4 requires the direct label 'Flux ET "
                "(Held Out)' 'rather than a separate Held-Out Evaluation heading or rule "
                "competing for space'",
                "the 'spread-weighted' label in the artwork; section 5.3 puts spread-dependent "
                "weighting in the caption by default",
                "the middle-dot record identification and its muted-gray subtitle treatment "
                "(section 5.1)",
                "the improvised '⊢—⊣' relation glyph in the E3 key; sections 4 and 5.4 write the "
                "relation with a plain em dash",
            ],
            "removed_strings": [
                "US-Bi1 · 2017",
                "Flux ET",
                "Held-Out Evaluation",
                "Inverse Estimation",
                "spread-weighted",
                "Daily State",
                "Daily Outputs",
                "Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3",
            ],
            "removed_strings_note": (
                "'US-Bi1 · 2017' -> 'US-Bi1 (2017)', near-black, no middle dot (section 5.1). "
                "'Flux ET' -> 'Flux ET (Held Out)': the merged ET region carries the held-out "
                "qualification on the trace itself, which is what lets the separate 'Held-Out "
                "Evaluation' label go (sections 5.2 item 5 and 5.4). 'Inverse Estimation' is "
                "retired outright: the closed cycle is named by its three stage labels 'Run "
                "Balance', 'Compare' and 'Update Parameters', and no section of the 2026-08-25 "
                "handoff asks for a title above them. 'spread-weighted' is caption-owned by "
                "section 5.3 and may return only on the incoming target route if a final-size "
                "proof shows it materially clarifies the member-spread encoding; that condition "
                "is recorded under inverse_cycle.spread_label_condition, not as a visible "
                "string. 'Daily State' -> 'State + Irrigation' and 'Daily Outputs' -> 'Daily ET' "
                "follow the section 4 wireframe row names for the two merged regions. The E3 "
                "relation string loses the improvised '⊢—⊣' glyph for the plain em dash written "
                "in sections 4 and 5.4. Retiring 'spread-weighted' also emptied "
                "CAPTION_PATTERN_EXEMPTIONS, so the r'\\bweight' caption pattern is back at full "
                "strength alongside r'σ'."
            ),
            "not_superseded": (
                "panel (b). Handoff section 6.1: 'Revision 3's core topology is accepted: the "
                "irrigated triangle is the fork, one horizontal leg reaches E2, and one visibly "
                "divergent curved leg reaches E3. Preserve that relationship.' Subsequent "
                "changes to arc curvature, map positions, clearances, or stroke weights are "
                "render-only unless they change the source or destination of a path."
            ),
        },
        "panel_a_redesign_2026_08_25": {
            "source": "paper/notes/fig01_production_handoff.md, rewritten 2026-08-25",
            "status_line": (
                "Revision 3 reviewed; panel (a) redesign and a new Gate A proof are required"
            ),
            "revision_level": (
                "Level 2 contract change (section 16): it changes visible copy, panel grouping, "
                "and inverse-estimation relationships"
            ),
            "changes": [
                {
                    "id": "r1_five_regions",
                    "sections": ["4", "5.2", "11", "13"],
                    "change": (
                        "panel (a) carries five quantitative plotting regions, not seven "
                        "independent sparklines: ETf Ensemble, NDVI Captures, Daily Forcing, "
                        "State + Irrigation, Daily ET"
                    ),
                },
                {
                    "id": "r2_merged_state_irrigation",
                    "sections": ["4", "5.2"],
                    "change": (
                        "root-zone depletion is the principal continuous trace and the "
                        "magnitude-bearing irrigation stems are integrated into the same "
                        "coordinated region, each with an explicit scale; a compact secondary "
                        "scale or clearly separated internal sub-band is preferable to "
                        "normalizing the stems"
                    ),
                },
                {
                    "id": "r3_merged_et",
                    "sections": ["4", "5.2", "5.4", "11"],
                    "change": (
                        "daily SWIM-RS ET and actual US-Bi1 flux ET are plotted together on ONE "
                        "axis with identical date support and vertical scale, directly labelled "
                        "'Daily ET' and 'Flux ET (Held Out)'"
                    ),
                },
                {
                    "id": "r4_closed_cycle",
                    "sections": ["4", "5.3", "10.3", "11", "13"],
                    "change": (
                        "inverse estimation becomes a literal compact closed cycle 'Run Balance "
                        "-> Compare -> Update Parameters -> Run Balance' with ETf targets and "
                        "'+ SWE' entering Compare and a 'Conditioned Parameters' exit leaving "
                        "the update stage toward the displayed daily trajectory"
                    ),
                },
                {
                    "id": "r5_y_axis_contract",
                    "sections": ["5.2", "9", "11", "13"],
                    "change": (
                        "every region reads as a quantitative small multiple: a short visible "
                        "left y-spine, at least two labelled bound values, units adjacent to the "
                        "axis or row label, and no clipped plotted value at the recorded display "
                        "limits"
                    ),
                },
                {
                    "id": "r6_near_black_identification",
                    "sections": ["5.1", "9", "11", "13"],
                    "change": (
                        "near-black #202124 for all reader-facing identification, titles, row "
                        "names, units, map count lines and inverse-stage labels; muted gray is "
                        "confined to axes, reference rules and quiet geographic context"
                    ),
                },
                {
                    "id": "r7_visible_copy",
                    "sections": ["5.1", "5.2", "5.3", "5.4"],
                    "change": (
                        "'US-Bi1 · 2017' -> 'US-Bi1 (2017)'; 'Flux ET' -> 'Flux ET (Held Out)'; "
                        "'Held-Out Evaluation', 'Inverse Estimation' and 'spread-weighted' "
                        "retired; 'Run Balance', 'Compare', 'Update Parameters' and 'Conditioned "
                        "Parameters' added"
                    ),
                },
                {
                    "id": "r8_panel_b_accepted",
                    "sections": ["6.1", "6.7"],
                    "change": (
                        "panel (b) is accepted as built in revision 3 and is not redesigned; the "
                        "faint CONUS state boundaries are retained and the HUC8 subdivisions are "
                        "kept out of the E3 map"
                    ),
                },
            ],
            "ambiguities_resolved": [
                {
                    "question": "does the 'Inverse Estimation' title survive?",
                    "resolution": "retired",
                    "basis": (
                        "the section 4 wireframe draws the cycle with no title above it; section "
                        "5.3 names only 'Run Balance', 'Compare' and 'Update Parameters'; "
                        "section 13 asks that 'inverse estimation is unmistakably a closed, "
                        "directional cycle through Run Balance, Compare, and Update Parameters'; "
                        "and section 9 speaks of 'inverse-stage labels', not an inverse title. "
                        "No section of the 2026-08-25 handoff requires the string, and section "
                        "5.3 asks the element to stay compact and unboxed, so a title would "
                        "spend space the redesign is trying to recover."
                    ),
                },
                {
                    "question": "what are the five region names?",
                    "resolution": (
                        "'ETf Ensemble', 'NDVI Captures', 'Daily Forcing', 'State + Irrigation', "
                        "'Daily ET'"
                    ),
                    "basis": (
                        "the section 4 wireframe row names are the only literal copy the handoff "
                        "gives; the section 5.2 item headings ('Water State and Irrigation', "
                        "'Daily ET and Held-Out Flux ET') are prose descriptions of the same "
                        "regions, and section 5.2 asks for compact facet labels rather than "
                        "prominent headings."
                    ),
                },
                {
                    "question": "where does the 'Conditioned Parameters' exit originate?",
                    "resolution": "the update stage",
                    "basis": (
                        "section 5.3: 'A separate exit labelled Conditioned Parameters leaves "
                        "the update stage after convergence and feeds the displayed daily state "
                        "and ET.' The section 4 wireframe hangs the exit under 'Run Balance'; "
                        "the prose is normative and is followed here."
                    ),
                },
                {
                    "question": "may 'spread-weighted' return?",
                    "resolution": (
                        "not as frozen copy; recorded as a conditional under "
                        "inverse_cycle.spread_label_condition"
                    ),
                    "basis": (
                        "section 5.3: 'It may return on the incoming target route only if a "
                        "final-size proof shows that it materially clarifies the member-spread "
                        "encoding.' No such proof exists, so the string is retired and its "
                        "caption-guard exemption removed."
                    ),
                },
                {
                    "question": "does the section 5.2 candidate ETf domain 0.8-1.6 hold?",
                    "resolution": (
                        "neither bound holds for the directed record: the floor is 0.0 and the "
                        "ceiling 1.8 (2026-08-27 revision for the S2 record)"
                    ),
                    "basis": (
                        f"the frozen {FIG01_EXAMPLE_SITE} record's retrieval members span "
                        "0.318-1.665, so a 0.8 floor would clip low member marks and the 1.6 "
                        "ceiling would clip the highest. Section 11 stops the build if a "
                        "plotted value is clipped at the recorded display limits, and section "
                        "5.2 calls its ranges 'candidate ranges for the next proof', so the "
                        "recorded domain widens to cover the data. (Under the superseded "
                        "US-Bi1 record the members spanned 0.152-1.452 and only the floor "
                        "moved.)"
                    ),
                },
            ],
        },
        "also_supersedes_3_1_0": {
            "schema_version": "3.1.0",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the second pass of the evidence-first redesign, rendered as the r2 proof. The "
                "user review of 2026-08-25 accepted the composition as a strong improvement and "
                "returned four must-address items plus refinements before Gate B: the held-out "
                "flux trace was labelled with the E1-E2 evaluation scope although the drawn "
                "record is a single US-Bi1 E1 series; the E3 key used an improvised 'Σ year' "
                "operator; 'Held-Out Evaluation' sat as a left-column secondary heading that "
                "opened a large empty region; the daily-ET-to-flux-ET comparison was drawn as a "
                "right-margin bracket that read as an empty box at thumbnail scale; the E3 "
                "transfer leg ran as a long horizontal corridor along the top of the E2 frame "
                "and read as an upper border on it, with the fork 2.8 mm ahead of the frame "
                "reading as part of the E2 entrance; and '+ SWE' sat beside the 'Inverse "
                "Estimation' title, where it read as part of the method name."
            ),
            "reason": (
                "User review of the r2 proof, 2026-08-25. Four must-address items and a set of "
                "refinements; the subset that changes frozen copy or frozen topology guidance is "
                "recorded here and in review_directives_2026_08_25."
            ),
            "removed_requirements": [
                "the 'Flux ET · E1–E2' label on the drawn US-Bi1 flux trace; the E1-E2 "
                "flux-evaluation scope is caption-owned",
                "the improvised 'Σ year' summation operator in the E3 evaluation key",
                "'Held-Out Evaluation' as a left-column secondary heading",
                "the drawn bracket or tie glyph on the daily-ET / flux-ET comparison",
                "the long horizontal E3 corridor paralleling the top of the E2 frame",
                "the '+ SWE' label placed adjacent to the 'Inverse Estimation' title",
            ],
            "removed_strings": [
                "Flux ET · E1–E2",
                "Σ year",
            ],
            "removed_strings_note": (
                "'Flux ET · E1–E2' became 'Flux ET': the drawn trace is an actual US-Bi1 record "
                "on the E1 cohort, so attaching the broader E1-E2 flux-evaluation role to that "
                "one series was a mislabel. The caption owns that scope, and caption_facts "
                "item_5_heldout still carries it. 'Σ year' became 'Annual Total': a true "
                "subscripted sigma would drop the 'year' subscript below the 7.5 pt floor, so "
                "the spelled-out operation wins. Retiring it also let the sigma exemption be "
                "removed from CAPTION_PATTERN_EXEMPTIONS, restoring the caption guard to full "
                "strength over every sigma-bearing string."
            ),
        },
        "review_directives_2026_08_25": {
            "source": "user review of the r2 proof, relayed 2026-08-25",
            "authority": (
                "paper/notes/fig01_production_handoff.md (revised 2026-08-24) remains the "
                "composition authority. These directives are layered on it and are cited here so "
                "the proof builder can trace each change to its origin."
            ),
            "gate": "one revision before Gate B",
            "directives": [
                {
                    "id": "d1_flux_label",
                    "kind": "frozen copy",
                    "change": "'Flux ET · E1–E2' -> 'Flux ET' (direct_label)",
                    "rationale": (
                        "the drawn trace is an actual US-Bi1 / E1 record, so the broader E1-E2 "
                        "flux-evaluation role was a mislabel on that series; the caption owns "
                        "the scope"
                    ),
                },
                {
                    "id": "d2_annual_total",
                    "kind": "frozen copy",
                    "change": (
                        "'Σ year' -> 'Annual Total' (direct_label); the E3 key now reads "
                        "'Daily Gross Applied Water → Annual Total ⊢—⊣ Metered Water · E3'"
                    ),
                    "rationale": (
                        "'Σ year' was improvised; a subscripted sigma would put 'year' below the "
                        "7.5 pt minimum, so the spelled-out operation is used instead"
                    ),
                },
                {
                    "id": "d3_heldout_label",
                    "kind": "frozen class and placement",
                    "change": "'Held-Out Evaluation' reclassified title -> direct_label",
                    "rationale": (
                        "as a left-column secondary heading it opened a large empty region; it "
                        "is now a compact label on the held-out rule itself"
                    ),
                },
                {
                    "id": "d4_comparison_alignment",
                    "kind": "frozen treatment",
                    "change": (
                        "the daily_et / flux_et comparison is shared alignment ONLY -- no drawn "
                        "bracket or tie glyph"
                    ),
                    "rationale": (
                        "the right-margin bracket read as an empty box at thumbnail scale; the "
                        "comparison entry itself is retained because the section 11 "
                        "common-date-mapping check depends on it, and the no-arrowhead rule "
                        "stands"
                    ),
                },
                {
                    "id": "d5_branch_geometry",
                    "kind": "frozen geometry guidance",
                    "change": (
                        "the irrigated-class triangle sits AT the fork with 'Irrigated "
                        "Parameters' set to its left; the E3 leg leaves the fork as a distinct "
                        "diagonal or shallow curve"
                    ),
                    "rationale": (
                        "the r2 horizontal corridor read as an upper border around the E2 frame, "
                        "and a junction 2.8 mm before the frame read as part of the E2 entrance; "
                        "the section 11 check is unchanged, this tightens the geometry it guards"
                    ),
                },
                {
                    "id": "d6_swe_placement",
                    "kind": "frozen placement",
                    "change": (
                        "'+ SWE' belongs on the incoming constraint route into the inverse "
                        "element, not beside the 'Inverse Estimation' title"
                    ),
                    "rationale": "beside the title it read as part of the method name",
                },
                {
                    "id": "d7_hillshade",
                    "kind": "open decision",
                    "change": (
                        "a faint privacy-safe E3 hillshade stays a user-optional future test; "
                        "not required for Gate B"
                    ),
                    "rationale": "it would need an archived, licensed DEM derivation first",
                },
            ],
        },
        "also_supersedes_3_0_0": {
            "schema_version": "3.0.0",
            "status": "review provenance, not an accepted composition",
            "superseded_composition": (
                "the first pass of the evidence-first redesign, rendered as "
                "paper/figures/proofs/fig01_evidence_190/. Panel (a) carried a narrow right-hand "
                "held-out column with the flux record compressed into a side strip beside the "
                "full-width model trace, an 'annual sum' bracket spanning the E1 example's "
                "seasonal applied-water record and reading as an E1-to-E3 data linkage, a "
                "three-circle data-like meter glyph with no frozen observations behind it, a "
                "separate SWE chip node, a 'PEST++ IES' subtitle on the inverse node, a faint "
                "driver lane that read as an axis spine, a second triangle irrigation lane in "
                "the daily-state row, and an E3 transfer path routed along the bottom of the E2 "
                "frame."
            ),
            "reason": (
                "fig01_production_handoff.md was revised again on 2026-08-24 after review of "
                "that proof. The evidence-first thesis is retained; the false E1-to-E3 data "
                "linkage, the held-out comparison grammar, the transfer routing, and the visual "
                "density are corrected."
            ),
            "removed_requirements": [
                "the narrow right-hand held-out region and the compressed flux side strip",
                "the 'annual sum' aggregation bracket over the E1 example's applied-water record",
                "the applied_water -> meters edge, which asserted a linkage the data do not "
                "support: E3 evaluates applied water simulated for E3 fields, not the E1 "
                "example",
                "the data-like three-circle meter glyph",
                "the separate swe_constraint chip node and its edge into inverse estimation",
                "the 'PEST++ IES' subtitle on the inverse-estimation node",
                "the second, triangle-based irrigation-event lane in the daily-state row",
                "the daily_et -> flux_et arrow; comparison is now a neutral tie",
                "row headings classified as titles competing with the panel heading",
            ],
            "removed_strings": [
                "PEST++ IES",
                "SWE",
                "annual sum",
                "Gross Applied Water",
                "irrigation",
                "2017",
                "Study Domains",
                "SWIM-RS Framework",
                "Satellite and Gridded Inputs",
                "NDVI · Vegetation State",
                "ETf + SWE · Calibration Targets",
                "Gridded Forcing · Model Drivers",
                "Daily Water Balance",
                "E1 · Source Parameters",
                "E2 · Geographic and Input Transfer",
                "E3 · Applied-Water Transfer",
                "E0 · Formulation Selection",
                "E2 · International",
            ],
            "removed_strings_note": (
                "The first six entries were visible in 3.0.0 and are removed by the 2026-08-24 "
                "revision; the rest were already removed when 3.0.0 superseded 2.1.0 and are "
                "carried forward so the revival guard still covers them. 'PEST++ IES' moved to "
                "the caption, 'SWE' became the inline '+ SWE' label, 'annual sum' became the "
                "'Σ year' operation in the separate E3 key, 'Gross Applied Water' became "
                "'Irrigation' on the E1 lane so it cannot be confused with the E3 key's 'Daily "
                "Gross Applied Water', 'irrigation' was the removed duplicate event lane's "
                "label, and '2017' was folded into 'US-Bi1 · 2017', which 3.2.0 in turn replaces "
                "with 'US-Bi1 (2017)'."
            ),
            "note": (
                "Architecture 3.0.0 and the proof under paper/figures/proofs/fig01_evidence_190/ "
                "are review provenance. Do not revise fig01_evidence_190.png into the next "
                "proof; write a fresh Gate A revision to a new proof directory (handoff sections "
                "1 and 15)."
            ),
        },
        "also_supersedes": {
            "schema_version": "2.1.0",
            "superseded_composition": (
                "map-plus-framework: panel (a) 'Study Domains', panel (b) 'SWIM-RS Framework' "
                "with a generic crop-soil cross-section and three grouped input cards, and one "
                "unlettered full-width transfer ribbon beneath both panels"
            ),
            "reason": (
                "fig01_production_handoff.md was rewritten 2026-08-20. The gray-box composition "
                "read as a graphical abstract: large labels, generic crop artwork, conceptual "
                "regions and unused space carried more weight than data. Figure 1 is now an "
                "evidence-first analytical figure whose density comes from real marks."
            ),
            "removed_requirements": [
                "the unlettered full-width transfer ribbon and its three ribbon nodes",
                "the generic crop-soil-water process cross-section",
                "the grouped input cards 'NDVI · Vegetation State', "
                "'ETf + SWE · Calibration Targets' and 'Gridded Forcing · Model Drivers'",
                "the 'Satellite and Gridded Inputs' input-region heading",
                "the 'Daily Water Balance' titled process node",
                "the inverse_estimation -> e1_source reading-path connector added 2026-08-20",
            ],
            "removed_strings": [
                "Study Domains",
                "SWIM-RS Framework",
                "Satellite and Gridded Inputs",
                "NDVI · Vegetation State",
                "ETf + SWE · Calibration Targets",
                "Gridded Forcing · Model Drivers",
                "Daily Water Balance",
                "E1 · Source Parameters",
                "E2 · Geographic and Input Transfer",
                "E3 · Applied-Water Transfer",
                "E0 · Formulation Selection",
                "E2 · International",
            ],
            "note": (
                "Version 2.1.0, the proofs under paper/figures/proofs/fig01_graybox_110/ and "
                "fig01_graybox/, and the assets under paper/figures/fig1_handoff/ are design "
                "provenance only (the retired scripts/figures/fig1_scope_architecture.py was "
                "removed from the repo 2026-09-22; see git history). They must not be "
                "polished, relabelled or reinserted."
            ),
        },
        "contract": (
            "paper/notes/fig01_production_handoff.md sections 4-9, rewritten 2026-08-25. Every "
            "string under panels / example_record / plotting_regions / driver_routing / "
            "inverse_cycle / outputs / held_out / e3_evaluation_key / map_nodes / "
            "parameter_tokens / development_tag / axes is frozen reader-facing copy and must be "
            "drawn verbatim. Nothing under caption_facts may be drawn."
        ),
        "canvas_mm": [190, 120],
        "outer_margin_mm": 3,
        "panel_gutter_mm": [3, 4],
        "canvas_note": (
            "Handoff section 4 (2026-08-20, unchanged 2026-08-24) sets 190 x 120 mm. "
            "six_figure_plan.md section 3.1 was reconciled to 120 mm on 2026-08-20; both notes "
            "now record the same decision."
        ),
        "figure_thesis": (
            "Sparse, disagreeing satellite observations condition a state-carrying daily water "
            "balance; its ET and applied-water outputs are evaluated with data that remain "
            "outside parameter estimation, and E1-derived parameter sets are tested through "
            "parallel transfer to E2 and E3."
        ),
        "reading_path": (
            "panel (a), top to bottom on one shared date axis, five quantitative plotting "
            "regions: the record identification, then the ETf ensemble captures, the NDVI "
            "captures, the daily forcing, the coordinated state-and-irrigation region in which "
            "the magnitude-bearing stems sit against the root-zone depletion trace, and finally "
            "the merged ET region in which the model trace and the held-out flux record share "
            "one axis, one date support and one vertical scale. Beneath them, one compact, "
            "unboxed closed cycle -- Run Balance to Compare to Update Parameters and back -- "
            "takes the acquisition-date ETf targets and the inline '+ SWE' constraint into "
            "Compare, is driven at Run Balance by NDVI and daily forcing, and leaves the update "
            "stage as a labelled 'Conditioned Parameters' exit into the displayed daily "
            "trajectory. A separate, typographic E3 evaluation key sits apart from the example "
            "record. Panel (b), left to right: the E1 CONUS source map, two class-specific "
            "parameter tokens, a visible branch junction at the irrigated token, then the E2 "
            "world map and the E3 San Luis Valley map as the two parallel transfer endpoints."
        ),
        "panels": [
            {
                "id": "panel_a",
                "letter": "(a)",
                "title": "Sparse Satellite Constraints to Daily State",
                "purpose": (
                    "make the model's temporal logic visible with actual data: intermittent "
                    "satellite information conditions the model while daily drivers and stored "
                    "soil water generate an uninterrupted trajectory and two outputs"
                ),
            },
            {
                "id": "panel_b",
                "letter": "(b)",
                "title": "E1 Source Cohort and Parallel Transfer",
                "purpose": (
                    "combine empirical geography and transfer topology in three coordinated "
                    "maps joined by class-specific parameter paths"
                ),
            },
        ],
        "example_record": {
            "file": "fig01_example_timeseries.csv",
            "selection_record": "fig01_example_selection.json",
            "site_id": FIG01_EXAMPLE_SITE,
            "window": [FIG01_EXAMPLE_START, FIG01_EXAMPLE_END],
            "n_days": FIG01_EXAMPLE_DAYS,
            "n_calibration_captures": FIG01_EXAMPLE_CAPTURES,
            "role": (
                "cross-figure visual motif shared with Figure 3. Figure 1 shows data form, "
                "cadence and state propagation only; it makes no performance claim and duplicates "
                "no part of Figure 3's benchmark comparison."
            ),
            "site_id_is_visible": True,
            "record_label": FIG01_RECORD_ID,
            "record_label_string_class": "direct_label",
            "record_label_placement": (
                "directly beneath the panel (a) heading, above the plotting regions; subordinate "
                "to the heading in weight but NOT in colour"
            ),
            "record_label_color": FIG01_IDENTIFICATION_COLOR,
            "record_label_rendered_as_muted_gray": False,
            "record_label_treatment": (
                "full black, set as identification rather than as a subtitle. Handoff section "
                "5.1 (stated there for the then-selected record; the treatment rules carry "
                f"over to the directed record {FIG01_RECORD_ID}): identify the record directly "
                "in the artwork in never-muted text; do not make the reader infer the site and "
                "year from the caption; avoid a muted gray subtitle and the middle-dot "
                "construction used in revisions 1-3. Hierarchy comes from size, weight and "
                "placement, never from washing the identification into gray (section 9)."
            ),
            "record_label_forbidden_treatments": [
                "muted gray secondary text",
                f"the middle-dot '{FIG01_EXAMPLE_SITE} · {FIG01_EXAMPLE_START[:4]}' "
                "construction of revisions 1-3",
                "supplying the site and year only in the caption",
            ],
            "site_id_note": (
                "handoff sections 5.1, 11 and 13 require the record to be identified in the "
                "artwork. The reader must not have to infer the site and year from the caption. "
                "Architecture 3.0.0 left this to the caption; that decision is reversed."
            ),
        },
        "shared_time_grammar": (
            "All five panel (a) plotting regions share one date axis. Sparse observations stay "
            "visibly discontinuous and are never connected or smoothed to look complete; daily "
            "quantities stay visibly continuous. Sparse ticks and units establish scale."
        ),
        "common_date_mapping": {
            "columns": list(FIG01_COMMON_DATE_MAPPING_COLUMNS),
            "axis": "date_axis",
            "region": "et_comparison",
            "lanes": ["daily_et", "flux_et"],
            "full_width": True,
            "comparable_horizontal_extent": True,
            "single_region": True,
            "shared_vertical_scale": True,
            "rule": (
                "the SWIM-RS ET trace and the held-out flux ET trace are drawn in ONE plotting "
                "region against one date mapping and one vertical scale: identical x limits, "
                "identical days-per-millimetre scale, identical horizontal extent spanning the "
                "full panel-(a) plotting width, and identical y limits. Handoff section 5.2 item "
                "5 (stated for the then-selected record): 'Plot daily SWIM-RS ET and actual "
                "flux ET together on one axis with "
                "identical date support and vertical scale.' They are presented as a direct "
                "visual comparison, so any difference in date or y mapping would misrepresent it "
                "(sections 5.2, 5.4, 10.3, 11 and 13)."
            ),
            "forbidden": [
                "compressing the flux record into a narrow right-hand side strip while the model "
                "trace runs full width",
                "a second, independent date axis for either trace",
                "a second, independent vertical scale for either trace",
                "separating the two traces into independent full-height lanes",
                "pairing an actual trace and a symbolic glyph inside one apparently equivalent "
                "evidence region",
                "an arrowhead directed into the flux observations",
                "an example metric, residual annotation, causal arrow, or separate comparison "
                "bracket",
            ],
            "comparison_treatment": (
                "shared alignment on one axis: one date mapping, one vertical scale, two direct "
                "labels. No bracket, tie glyph, or connector is drawn, and the relationship "
                "reads as a comparison, never as an arrow implying the model produced the "
                "observations."
            ),
        },
        "plotting_regions_note": (
            "Handoff section 4: 'Panel (a) contains five quantitative plotting regions, not "
            "seven independent sparklines. Root-zone depletion and irrigation share one "
            "coordinated region; daily SWIM-RS ET and held-out flux ET share another. Each "
            "region has an explicit vertical domain.' The seven-lane structure of architecture "
            "3.1.1 -- five numbered evidence rows with three lanes on the last -- is superseded, "
            "and the key is renamed from evidence_rows to plotting_regions so no downstream "
            "script can read the old grouping."
        ),
        "plotting_regions": [
            {
                "id": "etf_ensemble",
                "order": 1,
                "heading": "ETf Ensemble",
                "heading_string_class": "direct_label",
                "axis_label": "ETf",
                "axis_label_string_class": "direct_label",
                "columns": [f"etf_{m}" for m in E1_MEMBERS]
                + ["etf_target_mean", "etf_ensemble_spread", "etf_member_count"],
                "plotted_columns": etf_cols,
                "mark": (
                    f"at each of the {FIG01_EXAMPLE_CAPTURES} calibration captures, plot every available retrieval "
                    "member as a small neutral mark, a VISIBLE min-max line spanning the "
                    "members, and an OPEN target-mean diamond in satellite-benchmark orange. "
                    "Member disagreement must remain legible at final size and no member name is "
                    "drawn (handoff section 5.2 item 1)."
                ),
                "member_marks": "neutral",
                "min_max_line": True,
                "target_mean_marker": "open diamond",
                "color_role": "satellite_et_target",
                "member_names_visible": False,
                "y_axis": y_axis(
                    "etf_ensemble",
                    "ETf (dimensionless)",
                    [0.8, 1.6],
                    note=(
                        "the section 5.2 candidate range is 0.8-1.6. The frozen record's "
                        "retrieval members span 0.318-1.665, so a 0.8 floor would clip the low "
                        "member marks and the 1.6 ceiling the highest -- exactly the "
                        "disagreement this region exists to show; the recorded domain is "
                        "0.0-1.8 (2026-08-27, S2 record)."
                    ),
                ),
            },
            {
                "id": "ndvi_captures",
                "order": 2,
                "heading": "NDVI Captures",
                "heading_string_class": "direct_label",
                "axis_label": "NDVI",
                "axis_label_string_class": "direct_label",
                "columns": ["ndvi_landsat_raw", "ndvi_sentinel_raw"],
                "plotted_columns": ["ndvi_landsat_raw", "ndvi_sentinel_raw"],
                "mark": (
                    "raw Landsat and Sentinel-2 observations as distinct symbols, unconnected. "
                    "No daily filled NDVI trajectory is plotted; its fill provenance is "
                    "unresolved and the display file does not contain one."
                ),
                "filled_trace": False,
                "sensor_encoding": (
                    "shape first; the package sensor colors are used only if the two instruments "
                    "cannot be separated cleanly by shape alone"
                ),
                "y_axis": y_axis("ndvi_captures", "NDVI (dimensionless)", [0.2, 1.0]),
            },
            {
                "id": "daily_forcing",
                "order": 3,
                "heading": "Daily Forcing",
                "heading_string_class": "direct_label",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["eto", "precip"],
                "plotted_columns": ["eto", "precip"],
                "direct_labels": [
                    {"label": "ETo", "column": "eto", "string_class": "direct_label"},
                    {"label": "precipitation", "column": "precip", "string_class": "direct_label"},
                ],
                "mark": (
                    "compact precipitation bars and a fine ETo line on the shared date axis. "
                    "Handoff section 5.2 item 3: 'This is one quantitative plot, not a product "
                    "inventory.'"
                ),
                "single_quantitative_plot": True,
                "y_axis": y_axis("daily_forcing", "mm d⁻¹", [0, "rounded maximum"]),
            },
            {
                "id": "state_and_irrigation",
                "order": 4,
                "heading": "State + Irrigation",
                "heading_string_class": "direct_label",
                "axis_label": "mm",
                "axis_label_string_class": "direct_label",
                "columns": ["rz_depletion", "irr_applied"],
                "plotted_columns": ["rz_depletion", "irr_applied"],
                "merged": True,
                "merge_note": (
                    "handoff section 5.2 item 4, 'Water State and Irrigation': root-zone "
                    "depletion is the principal continuous trace and the magnitude-bearing "
                    "simulated irrigation stems are integrated into the SAME coordinated region. "
                    "'This replaces the generic plant-soil block and makes event-to-state "
                    "response visible without consuming a separate full-height lane.' "
                    "Architecture 3.1.1 kept them in two separate regions; that split is "
                    "superseded."
                ),
                "direct_labels": [
                    {
                        "label": "root-zone depletion",
                        "column": "rz_depletion",
                        "string_class": "direct_label",
                    },
                    {
                        "label": "Irrigation",
                        "column": "irr_applied",
                        "node": "applied_water",
                        "string_class": "direct_label",
                    },
                ],
                "principal_trace": {
                    "id": "rz_depletion",
                    "column": "rz_depletion",
                    "mark": (
                        "root-zone depletion as one continuous line or band, vertically aligned "
                        "with the precipitation bars above. This is the primary visual evidence "
                        "of state propagation and replaces the removed generic plant-soil block."
                    ),
                    "color_role": "swim_state_and_output",
                },
                "irrigation_stems": {
                    "id": "applied_water",
                    "column": "irr_applied",
                    "mark": (
                        "magnitude-bearing applied-water stems: one stem per simulated "
                        "irrigation event, its height carrying the daily depth on its own "
                        "explicit scale. This is the figure's only irrigation encoding."
                    ),
                    "color_role": "swim_state_and_output",
                    "scope": f"the E1 {FIG01_EXAMPLE_SITE} example only",
                    "not_an_e3_series": True,
                    "normalized_to_state_scale": False,
                    "scale_treatment": (
                        "a compact secondary scale or a clearly separated internal sub-band "
                        "inside this region. Handoff section 5.2 item 4 prefers either to "
                        "normalizing the stems, so the stem heights must stay readable as "
                        "millimetres of applied water rather than as a unitless fraction of the "
                        "depletion axis."
                    ),
                    "forbidden": [
                        "normalizing the stems to the depletion scale",
                        "any aggregation bracket, tie, or sum drawn over these stems",
                        "any drawn connection to the E3 evaluation key or its Annual Total "
                        "operation",
                        "repeating these events as triangles or another symbol lane anywhere in "
                        "panel (a)",
                    ],
                },
                "irrigation_events_repeated_here": False,
                "irrigation_note": (
                    "one irrigation encoding only (handoff section 5.2). The stems carry both "
                    "timing and magnitude, and now sit in the same region as the state trace, so "
                    "event-to-state response is read directly rather than across two lanes."
                ),
                "color_role": "swim_state_and_output",
                "y_axis": y_axis(
                    "state_and_irrigation",
                    "mm (root-zone depletion)",
                    [0, "rounded maximum"],
                ),
                "secondary_y_axis": y_axis(
                    "state_and_irrigation_secondary",
                    "mm (applied water)",
                    [0, "rounded maximum"],
                    note=(
                        "the explicit second scale handoff section 5.2 item 4 requires for the "
                        "irrigation magnitudes; without it the stems could only be drawn "
                        "normalized, which the section rules out"
                    ),
                ),
            },
            {
                "id": "et_comparison",
                "order": 5,
                "heading": "Daily ET",
                "heading_string_class": "direct_label",
                "axis_label": "mm d⁻¹",
                "axis_label_string_class": "direct_label",
                "columns": ["swim_ET", "flux_ET"],
                "plotted_columns": ["swim_ET", "flux_ET"],
                "merged": True,
                "merge_note": (
                    "handoff section 5.2 item 5, 'Daily ET and Held-Out Flux ET' (stated for "
                    "the then-selected record): 'Plot daily "
                    "SWIM-RS ET and actual flux ET together on one axis with identical "
                    "date support and vertical scale. Directly label the lines Daily ET and Flux "
                    "ET (Held Out). Do not add an example metric, residual annotation, causal "
                    "arrow, or a separate comparison bracket.' The two traces are lanes of ONE "
                    "region, not two full-height lanes."
                ),
                "shared_axis": True,
                "shared_vertical_scale": True,
                "lanes": [
                    {
                        "id": "daily_et",
                        "label": "Daily ET",
                        "string_class": "direct_label",
                        "column": "swim_ET",
                        "mark": "a continuous daily model trace at full panel width",
                        "color_role": "swim_state_and_output",
                        "horizontal_extent": "full panel-(a) plotting width",
                    },
                    {
                        "id": "flux_et",
                        "label": "Flux ET (Held Out)",
                        "string_class": "direct_label",
                        "column": "flux_ET",
                        "label_note": (
                            "3.2.0: handoff section 5.4 requires 'the direct label Flux ET (Held "
                            "Out) rather than a separate Held-Out Evaluation heading or rule "
                            "competing for space'. The qualification now travels with the trace, "
                            "which is what lets the 'Held-Out Evaluation' label be retired. The "
                            "broader E1-E2 flux-evaluation scope stays caption-owned "
                            "(caption_facts item_5_heldout)."
                        ),
                        "mark": (
                            "the actual example-site flux record as a thin near-black trace, drawn on "
                            "the SAME axis, the SAME date mapping and the SAME vertical scale as "
                            "the model trace"
                        ),
                        "color_role": "held_out_observation",
                        "horizontal_extent": "full panel-(a) plotting width, identical to daily_et",
                        "held_out": True,
                        "comparison_treatment": (
                            "shared alignment only -- one axis, one date mapping, one vertical "
                            "scale, two direct labels. No bracket, tie glyph, or connector is "
                            "drawn, no arrowhead may point into this lane, and no example "
                            "metric, residual, or emphasis on agreement may accompany it"
                        ),
                    },
                ],
                "aggregation_mark": None,
                "aggregation_note": (
                    "annual aggregation appears only in the separate, experiment-level "
                    "e3_evaluation_key. No bracket, tie, or sum may span this region or the "
                    "irrigation stems above it (handoff sections 4, 5.2, 5.4 and 9)."
                ),
                "color_role": "swim_state_and_output",
                "y_axis": y_axis("et_comparison", "mm d⁻¹", [0, 10]),
            },
        ],
        "plotting_region_rules": [
            "exactly five regions; handoff section 4 forbids seven independent sparklines and "
            "section 11 stops the build above five",
            "every region reads as a quantitative small multiple rather than a sparkline: a "
            "short visible left y-spine, at least two labelled values defining the displayed "
            "lower and upper bounds, and units adjacent to the axis or the region label",
            "no plotted value may be clipped at the recorded display limits",
            "the five region names are compact facet or y-axis labels, not prominent display "
            "headings; one shared date axis serves all five",
            "rounded, stable domains are preferred; axis-range changes derived from the same "
            "frozen columns are render-level settings, not a reason to rebuild data",
        ],
        "y_axis_contract": {
            "source": "handoff sections 5.2, 9, 11 and 13",
            "required_per_region": [
                "a short visible left y-spine",
                "at least two labelled bound values",
                "units adjacent to the axis or the region label",
                "no clipped plotted value at the recorded display limits",
            ],
            "spine_weight_pt": [0.45, 0.7],
            "final_limits_are_render_level": (
                "the display limits recorded here are the contract the section 11 clipping check "
                "runs against. A proof may refine ticks, padding and label placement inside them "
                "as a Level 1 render change; widening or narrowing them past a plotted value is "
                "not permitted"
            ),
            "candidate_domains_2026_08_25": {
                "etf": [0.8, 1.6],
                "ndvi": [0.2, 1.0],
                "forcing": [0, "rounded maximum"],
                "depletion": [0, "rounded maximum"],
                "irrigation": [0, "rounded maximum"],
                "et": [0, 10],
            },
            "recorded_display_domains": {
                k: list(v) for k, v in sorted(FIG01_DISPLAY_DOMAINS.items())
            },
            "observed_data_ranges": {k: v for k, v in sorted(region_ranges.items())},
        },
        "date_axis": {
            "id": "date_axis",
            "label": None,
            "string_class": "direct_label",
            "columns": ["date"],
            "treatment": (
                "one shared date axis for all five plotting regions, drawn once at the bottom of "
                "panel (a) with sparse month ticks; tick text is generated from the data and is "
                "not frozen copy"
            ),
            "label_note": (
                "the standalone year axis label of architecture 3.0.0 is dropped: the year is "
                f"now carried by the required record identification '{FIG01_RECORD_ID}', and "
                "repeating it on the axis would spend a visible string on a fact already stated"
            ),
        },
        "inverse_cycle": {
            "id": "inverse_cycle",
            "title": None,
            "title_string_class": "direct_label",
            "title_note": (
                "3.2.0: the 'Inverse Estimation' title of 3.1.1 is RETIRED. The section 4 "
                "wireframe draws the cycle with no title above it, section 5.3 names only the "
                "three stages, section 13 asks that 'inverse estimation is unmistakably a "
                "closed, directional cycle through Run Balance, Compare, and Update Parameters', "
                "and section 9 speaks of 'inverse-stage labels'. The three stage labels name the "
                "element; a separate title would spend space the redesign is recovering and is "
                "held under the revival guard."
            ),
            "representation": "literal directed cycle",
            "compact_treatment": (
                "keep the cycle compact and unboxed: a small triangular or circular routing of "
                "the three verbs is sufficient. No process card, no fill, no thick rounded box, "
                "no drop shadow (handoff sections 5.3, 9 and 13)."
            ),
            "boxed": False,
            "stages": [
                {
                    "id": "run_balance",
                    "order": 1,
                    "label": "Run Balance",
                    "string_class": "direct_label",
                    "role": "the balance run produces simulated calibration quantities",
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
                {
                    "id": "compare",
                    "order": 2,
                    "label": "Compare",
                    "string_class": "direct_label",
                    "role": (
                        "comparison of simulated calibration quantities with the acquisition-date "
                        "ETf targets and the auxiliary SWE constraint produces mismatches"
                    ),
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
                {
                    "id": "update_parameters",
                    "order": 3,
                    "label": "Update Parameters",
                    "string_class": "direct_label",
                    "role": "the parameter update feeds the next balance run",
                    "color": FIG01_IDENTIFICATION_COLOR,
                },
            ],
            "cycle_edges": [list(e) for e in FIG01_CYCLE_EDGES],
            "cycle_rule": (
                "handoff section 5.3: 'Arrowheads must close the cycle in that order: the "
                "balance produces simulated calibration quantities, comparison with targets "
                "produces mismatches, and the parameter update feeds the next balance run.' The "
                "internal feedback edge belongs only to simulated calibration quantities."
            ),
            "constraint_inputs": [
                {
                    "from": "etf_ensemble",
                    "into": FIG01_CONSTRAINT_INTO,
                    "drawn_as_edge": True,
                    "note": "acquisition-date ETf targets enter the comparison stage",
                },
                {
                    "from": "swe_inline",
                    "into": FIG01_CONSTRAINT_INTO,
                    "drawn_as_edge": False,
                    "note": (
                        "'+ SWE' rides the same incoming constraint route as an inline label; it "
                        "is not a separate node, card, or arrow, and it is not a zero-valued "
                        "trace in this growing-season example"
                    ),
                },
            ],
            "driver_inputs": [
                {"from": "ndvi_captures", "into": FIG01_DRIVER_INTO, "drawn_as_edge": True},
                {"from": "daily_forcing", "into": FIG01_DRIVER_INTO, "drawn_as_edge": True},
            ],
            "exit": {
                "id": "conditioned_parameters",
                "label": FIG01_CYCLE_EXIT_LABEL,
                "string_class": "direct_label",
                "edge": list(FIG01_CYCLE_EXIT_EDGE),
                "drawn_as": "a labelled exit arrow, not a fourth stage box",
                "color": FIG01_IDENTIFICATION_COLOR,
                "rule": (
                    "handoff section 5.3: 'A separate exit labelled Conditioned Parameters "
                    "leaves the update stage after convergence and feeds the displayed daily "
                    "state and ET.' The section 4 wireframe hangs the exit under Run Balance; "
                    "the prose is normative and is followed here."
                ),
            },
            "routing_rule": (
                "the inverse path and the driver path must be distinguishable through routing "
                "and stroke treatment. ETf and SWE constrain parameters and enter Compare; NDVI "
                "and daily forcing drive the forward balance and enter Run Balance. Route the "
                "driver connector away from y-axis spines and tick labels so it cannot be "
                "mistaken for an axis. The figure must not imply that all inputs share the same "
                "dates or enter through the same mechanism."
            ),
            "held_out_rule": (
                "flux ET and meter observations remain outside the cycle and have no return "
                "path. Handoff section 10.3: 'Held-out flux and meter observations remain "
                "forbidden as cycle inputs.'"
            ),
            "spread_label_condition": {
                "string": "spread-weighted",
                "status": "retired from the artwork; caption-owned",
                "may_return": True,
                "return_condition": (
                    "handoff section 5.3: 'It may return on the incoming target route only if a "
                    "final-size proof shows that it materially clarifies the member-spread "
                    "encoding.' No such proof exists."
                ),
                "return_placement": "the incoming target route only",
                "return_requires": (
                    "restoring a CAPTION_PATTERN_EXEMPTIONS entry for the exact string, since "
                    "r'\\bweight' otherwise blocks it as caption-owned copy"
                ),
                "recorded_as": "a conditional, not a visible string",
            },
            "engine_label_ownership": "caption",
            "engine_label_condition": (
                "'PEST++ IES' stays in the caption by default (handoff section 5.3). The engine "
                "name is carried by caption_facts.calibration_settings and by the working "
                "caption."
            ),
            "color_role": "inverse_estimation",
            "label_color_rule": (
                "structural cycle labels remain near-black even when their arrows are purple "
                "(handoff section 9)"
            ),
            "forbidden_content": [
                "a title above the cycle",
                "the objective equation",
                "the parameter inventory",
                "the realization count",
                "the iteration count",
                "the weighting formula",
                "an engine name that forces the element to grow",
            ],
            "forbidden_representations": [
                "two disconnected L-shaped legs sharing a heading",
                "a generic one-way transformation",
                "a large process card, a boxed node, or a tinted region",
                "a separate SWE card, chip, or arrow",
                "a zero-valued SWE trace",
                "any edge from flux ET or meter observations into any stage",
            ],
            "supersedes": (
                "the one-way inverse representation of architectures 3.0.0-3.1.1: a titled "
                "'Inverse Estimation' element whose iteration was implied by a single "
                "inverse_estimation -> daily_balance parameter-update arrow. Handoff sections "
                "5.3, 10.3 and 11 now require the literal closed cycle and the labelled exit."
            ),
        },
        "swe_constraint": {
            "id": "swe_inline",
            "label": "+ SWE",
            "string_class": "direct_label",
            "column": "swe_audit",
            "rendered_as": (
                "inline label on the incoming constraint route into the 'Compare' stage (3.2.0)"
            ),
            "enters_stage": FIG01_CONSTRAINT_INTO,
            "drawn_as_separate_node": False,
            "drawn_as_edge": False,
            "placement": (
                "on the constraint route entering the 'Compare' stage of the inverse cycle, "
                "alongside the acquisition-date ETf targets. Handoff section 5.3: "
                "'Acquisition-date ETf targets and the inline + SWE auxiliary constraint enter "
                "the Compare stage.' The r2 proof's placement beside the retired 'Inverse "
                "Estimation' title made it read as part of the method name."
            ),
            "treatment": (
                "gridded SWE is an auxiliary calibration constraint written inline on the route "
                "into the compare stage, not a separate card, chip, or arrow. It is identically "
                "zero across this growing-season example, so no zero-valued SWE trace may be "
                "forced into the panel (handoff section 5.3)."
            ),
            "plotted_as_trace": False,
            "supersedes": (
                "the 'SWE' chip node and the swe_constraint -> inverse_estimation edge of "
                "architecture 3.0.0, and the 3.1.1 placement on the route into the titled "
                "inverse element"
            ),
        },
        "driver_routing": {
            "id": "daily_drivers",
            "label": "daily drivers",
            "string_class": "direct_label",
            "applies_to": ["ndvi_captures", "daily_forcing"],
            "destination": FIG01_DRIVER_INTO,
            "optional": True,
            "treatment": (
                "a short direct label on the driver connector, in place of the faint full-height "
                "lane drawn by architecture 3.0.0. Handoff section 5.3: 'If retained, the daily "
                "drivers bracket should span only the NDVI and forcing rows and use one short "
                "connector into Run Balance; it must not become a long non-data stroke.'"
            ),
            "spans_regions": ["ndvi_captures", "daily_forcing"],
            "layout_constraint": (
                "route the driver connector clear of the y-axis spines, the ticks and the tick "
                "labels so it cannot be mistaken for an axis; it must not overlap or run "
                "parallel and adjacent to any axis line (handoff sections 5.3 and 13)"
            ),
            "forbidden": [
                "a faint vertical lane running alongside the region spines",
                "any routing that overlaps axis tick text",
                "a stroke weight or colour that matches the axes",
                "a bracket spanning regions other than NDVI Captures and Daily Forcing",
                "a long non-data stroke",
            ],
        },
        "daily_balance": {
            "id": "daily_balance",
            "represented_by": ["state_and_irrigation", "et_comparison"],
            "label": None,
            "note": (
                "the displayed daily trajectory has no titled box. It is represented by the "
                "actual State + Irrigation and Daily ET plotting regions, and it is reached by "
                "the labelled 'Conditioned Parameters' exit from the update stage. The removed "
                "generic crop-soil cross-section must not be restored."
            ),
            "relation_to_run_balance": (
                "'Run Balance' inside the cycle and this displayed trajectory are the same "
                "forward water balance in two roles: iterated while parameters are conditioned, "
                "then displayed once conditioned. The drivers therefore enter at Run Balance and "
                "are not drawn twice."
            ),
        },
        "outputs": [
            {
                "id": "daily_et",
                "label": "Daily ET",
                "string_class": "direct_label",
                "column": "swim_ET",
                "region": "et_comparison",
                "compared_with": "flux_et",
                "comparison_kind": "shared alignment on one axis; not a directed edge",
            },
            {
                "id": "applied_water",
                "label": "Irrigation",
                "string_class": "direct_label",
                "column": "irr_applied",
                "region": "state_and_irrigation",
                "compared_with": None,
                "aggregation": None,
                "scope": f"the E1 {FIG01_EXAMPLE_SITE} example only",
                "note": (
                    "renamed from 'Gross Applied Water' so it cannot be confused with the "
                    "separate E3 key's 'Daily Gross Applied Water'. This series has NO drawn "
                    "comparison and NO drawn aggregation: E3 evaluates applied water simulated "
                    "for E3 fields, not this record. The applied_water -> meters edge of "
                    "architecture 3.0.0 is removed and is now a forbidden edge (handoff sections "
                    "5.2, 5.4 and 11)."
                ),
            },
        ],
        "held_out": {
            "heading": None,
            "heading_string_class": "direct_label",
            "heading_treatment": (
                "3.2.0: there is NO held-out heading. Handoff section 5.4 requires 'the direct "
                "label Flux ET (Held Out) rather than a separate Held-Out Evaluation heading or "
                "rule competing for space'. The qualification now travels with the trace inside "
                "the merged ET region, so the 3.1.1 label is retired and held under the revival "
                "guard."
            ),
            "heading_pt": None,
            "divider": (
                "no rule separates the two ET traces -- they share one axis. A quiet rule may "
                "still set off the separate E3 evaluation key. Nothing about the held-out "
                "boundary is two-way: no return path, no evaluation-to-fitting arrow, and no "
                "arrowhead directed into either observation."
            ),
            "region_treatment": (
                "the two held-out treatments stay separated rather than squeezed into one narrow "
                "right-hand column: the flux record is a lane of the merged ET region at full "
                "panel width, and the E3 meter term is a typographic node in the separate key. "
                "No tinted card may be placed around either."
            ),
            "narrow_right_hand_column": False,
            "supersedes": (
                "architecture 3.0.0 placed both treatments in a narrow right-hand region with "
                "the flux record compressed into a side strip beside the full-width model trace, "
                "and paired that actual trace with a symbolic meter glyph inside one apparently "
                "equivalent evidence region. Handoff section 5.4 forbids both."
            ),
            "observations": [
                {
                    "id": "flux_et",
                    "label": "Flux ET (Held Out)",
                    "string_class": "direct_label",
                    "aligned_with": "daily_et",
                    "region": "et_comparison",
                    "lane": "flux_et",
                    "column": "flux_ET",
                    "is_actual_data": True,
                    "treatment": (
                        "the actual example-site flux record drawn as a thin near-black trace at full "
                        "panel width, on the SAME axis, date mapping and vertical scale as the "
                        "Daily ET trace. Comparison is by shared alignment ONLY: no bracket, no "
                        "tie glyph, no connector, no arrow into the trace, no example metric, no "
                        "residual, and no emphasis on agreement."
                    ),
                    "scope_note": (
                        "the label is 'Flux ET (Held Out)' (3.2.0, handoff sections 5.2 item 5 "
                        "and 5.4): the held-out qualification travels with the trace instead of "
                        "occupying a separate heading. This is one E1 site-year; the broader "
                        "E1-E2 flux-evaluation scope is caption-owned."
                    ),
                },
                {
                    "id": "meters",
                    "label": "Metered Water · E3",
                    "string_class": "direct_label",
                    "aligned_with": "e3_annual_sum",
                    "lane": None,
                    "column": None,
                    "is_actual_data": False,
                    "data_like_glyph": False,
                    "frozen_observation_source": None,
                    "treatment": (
                        "a typographic term in the separate, experiment-level E3 evaluation key "
                        "-- not a mark inside the E1 example. No E3 meter values are frozen "
                        "in the Figure 1 package, so nothing here may be drawn as data. The "
                        "three-circle meter glyph of the reviewed fig01_evidence_190 proof read "
                        "as data and is disallowed. Real meter marks require frozen observations "
                        "selected by an observation-support rule independent of simulated "
                        "performance, plus the corresponding E3 model output on an honest common "
                        "support, plus its own selection record (handoff section 5.4)."
                    ),
                },
            ],
            "color_role": "held_out_observation",
        },
        "e3_evaluation_key": {
            "id": "e3_key",
            "scope": (
                "experiment-level. It describes the E3 evaluation operation on applied water "
                "simulated for E3 fields. It is NOT an extension of the E1 example record and "
                "shares no mark, tie, bracket, or connector with it."
            ),
            "treatment_class": "typographic_schematic",
            "placement": (
                "below the panel (a) plotting regions, visibly separated from the shared date "
                "axis and from the irrigation stems, so no reader can trace a path from the E1 "
                "stems into it"
            ),
            "relation_string": "Daily Gross Applied Water → Annual Total — Metered Water · E3",
            "relation_string_note": (
                "3.1.1 (review directive d2) replaced the improvised 'Σ year' with the "
                "spelled-out 'Annual Total'. 3.2.0 also drops the improvised '⊢—⊣' tie glyph for "
                "the plain em dash written in handoff sections 4 and 5.4: 'Daily Gross Applied "
                "Water → Annual Total — Metered Water · E3'. The comparison to the meters stays "
                "a neutral, undirected relation -- no arrowhead into the observation."
            ),
            "nodes": [
                {
                    "id": "e3_applied_water",
                    "label": "Daily Gross Applied Water",
                    "string_class": "direct_label",
                    "column": None,
                    "source": "E3 simulations; no values are frozen in the Figure 1 package",
                    "is_actual_data": False,
                },
                {
                    "id": "e3_annual_sum",
                    "label": "Annual Total",
                    "string_class": "direct_label",
                    "column": None,
                    "is_actual_data": False,
                    "purpose": (
                        "the compact annual-aggregation operation required by handoff sections 9 "
                        "and 13, in place of a bracket spanning an entire seasonal record"
                    ),
                    "label_note": (
                        "3.1.1 (review directive d2): replaces the improvised 'Σ year'. Because "
                        "no sigma is drawn, no caption-guard exemption is needed and the sigma "
                        "pattern is back at full strength."
                    ),
                },
                {
                    "id": "meters",
                    "label": "Metered Water · E3",
                    "string_class": "direct_label",
                    "column": None,
                    "is_actual_data": False,
                },
            ],
            "internal_edges": [["e3_applied_water", "e3_annual_sum"]],
            "internal_comparisons": [["e3_annual_sum", "meters"]],
            "data_like_glyph": False,
            "frozen_observation_source": None,
            "forbidden": [
                "the three-circle meter glyph, or any other data-like meter mark, while no E3 "
                "observations are frozen",
                "any connection to the E1 example's Irrigation stems in the State + Irrigation "
                "region, or to a sum of them",
                "an arrowhead directed into 'Metered Water · E3'",
                "drawing this key inside the example's plotting region or on its date axis",
            ],
            "upgrade_path": (
                "to draw real E3 marks, first freeze the E3 modelled and observed values with an "
                "independently justified selection record and display them on an honest common "
                "support; do not extend fig01_example_timeseries.csv to cover them"
            ),
        },
        "map_nodes": [
            {
                "id": "e1_map",
                "heading": "E1 · CONUS",
                "heading_string_class": "title",
                "count_line": "60 Cropland Sites",
                "count_line_string_class": "direct_label",
                "layer": "e1_sites",
                "rows": 60,
                "context_layer": "conus_context",
                "optional_context_layer": "conus_states_context",
                "optional_context_layer_status": (
                    "ACCEPTED for 3.2.0. Handoff section 6.7: 'Retain the faint CONUS state "
                    "boundaries accepted in revision 3.' Draw them as hairline neutral "
                    "boundaries, strictly subordinate to the site marks; no state labels, no "
                    "fills."
                ),
                "width_mm": [48, 52],
                "color_role": "e1",
                "treatment": (
                    "plot all 60 configured sites including MB_Pch on a quiet CONUS base with "
                    "enough internal geographic context to orient the reader; do not label "
                    "individual towers"
                ),
            },
            {
                "id": "e2_map",
                "heading": "E2 · 10 Countries",
                "heading_string_class": "title",
                "count_line": "66 Cropland Sites",
                "count_line_string_class": "direct_label",
                "layer": "e2_sites",
                "rows": 66,
                "context_layer": "world_context",
                "width_mm": [65, 72],
                "color_role": "e2",
                "treatment": (
                    "plot all 66 configured sites across ten countries and four continents; crop "
                    "empty polar latitudes and use a projection that gives the occupied "
                    "continents adequate area; one quiet boundary hierarchy, no choropleth. The "
                    "reader should perceive reach and clustering, not count sites."
                ),
            },
            {
                "id": "e3_map",
                "heading": "E3 · San Luis Valley",
                "heading_string_class": "title",
                "count_line": "50 Metered Fields",
                "count_line_string_class": "direct_label",
                "layer": "e3_display",
                "rows": 50,
                "context_layer": "slv_context",
                "optional_context_layer": None,
                "optional_context_layer_status": (
                    "REJECTED for 3.2.0. Handoff section 6.7: 'Keep the HUC8 subdivisions out of "
                    "the E3 map; they competed with the field marks and implied an analytical "
                    "spatial grouping that E3 does not use.' The slv_basin_context layer stays "
                    "in fig01_scope.gpkg as archived, licensed, hashed provenance -- removing it "
                    "would be a Level 3 geography change for no gain -- but it must not be "
                    "drawn. The generalized San Luis Valley boundary (slv_context) is retained."
                ),
                "width_mm": [30, 36],
                "color_role": "e3",
                "keep_on_one_line": "San Luis Valley",
                "treatment": (
                    "plot the 50 primary metered fields as the approved generalized display "
                    "(centroids snapped to a 1 km EPSG:5070 grid, reprojected for display). No "
                    "exact field polygons, source-agency identifiers, acreage, or "
                    "source-to-display linkage. If linked to the E1 locator, use a fine magenta "
                    "leader that survives grayscale."
                ),
            },
        ],
        "map_overlap_note": (
            "13 of the 66 E2 sites also occur in E1. Handoff section 6.2 keeps that fact in the "
            "caption: no additional overlap map symbol is drawn unless a final-size proof shows "
            "it improves rather than obscures the transfer story."
        ),
        "parameter_tokens": [
            {
                "id": "irrigated_params",
                "label": "Irrigated Parameters",
                "string_class": "direct_label",
                "source": "e1_map",
                "destinations": ["e2_map", "e3_map"],
                "symbol": "triangle",
                "symbol_placement": (
                    "3.1.1 (review directive d5): the triangle sits AT the fork itself -- the "
                    "point where the E2 and E3 legs separate -- with the 'Irrigated Parameters' "
                    "label set to its LEFT. The symbol marks the junction, so the split is read "
                    "as originating at the class token and not at either destination frame."
                ),
            },
            {
                "id": "rainfed_params",
                "label": "Rainfed Parameters",
                "string_class": "direct_label",
                "source": "e1_map",
                "destinations": ["e2_map"],
                "symbol": "circle",
            },
        ],
        "parameter_token_treatment": (
            "two compact tokens or parallel paths occupying roughly 15-20 mm between the E1 map "
            "and the destination maps, visually subordinate to the maps. They encode direction "
            "and class only. No parameter names, values, priors, or internal run identifiers."
        ),
        "transfer_branch": {
            "id": "irrigated_branch",
            "at": FIG01_TRANSFER_BRANCH_AT,
            "visible": True,
            "outgoing": ["e2_map", "e3_map"],
            "position": (
                "AT the Irrigated Parameters token, marked by its triangle, and well BEFORE the "
                "E2 map frame, so both irrigated paths are seen to originate at the token. A "
                "junction placed a couple of millimetres ahead of the frame reads as part of the "
                "E2 entrance and is not acceptable (3.1.1, review directive d5: the r2 fork sat "
                "2.8 mm before the frame and read that way)."
            ),
            "e3_route": (
                "a clearly separate leg leaving the fork as a diagonal or shallow curve to the "
                "E3 map, visibly diverging from the E2 leg at the junction. It must NOT run as a "
                "long horizontal corridor paralleling the top or bottom of the E2 frame -- in "
                "the r2 proof that corridor read as an upper border drawn around E2. It may "
                "stagger vertically or pass above or below the E2 map, but it must never run "
                "along, beneath, or through the E2 frame in a way that can be read as E2->E3."
            ),
            "forbidden_routes": [
                "the reviewed fig01_evidence_190 route, which ran along the bottom of the E2 "
                "frame and read as E2 -> E3",
                "the r2 route, a long horizontal corridor along the top of the E2 frame that "
                "read as an upper border on the E2 map",
                "any route whose visible origin is the E2 map or its frame",
                "any route that enters and leaves the E2 map's plotting region",
                "any fork placed close enough to the E2 frame to read as its entrance",
            ],
            "rule": (
                "handoff sections 6.1 and 6.3: make the irrigated split explicit with one branch "
                "point immediately after its token; both outgoing paths originate there and "
                "neither originates at the E2 map"
            ),
        },
        "symbol_encoding": {
            "attribute": "irrigation_class",
            "source": "frozen E1/E2 source-class assignment used to construct the parameter "
            "sets, not annual irrigation activation",
            "shapes": {"irrigated": "triangle", "rainfed": "circle"},
            "all_e3_fields_use": "triangle",
            "all_e3_fields_reason": (
                "E3 evaluates only the irrigated transfer set, so every E3 field carries the "
                "irrigated shape"
            ),
            "color_encodes": "experiment, not class",
            "legend": (
                "none; the two parameter-token labels carry the class names and the shapes are "
                "read from them"
            ),
            "frozen_counts": {
                "e1_irrigated": FIG01_CLASS_COUNTS["e1_irrigated"],
                "e1_rainfed": FIG01_CLASS_COUNTS["e1_rainfed"],
                "e2_irrigated": FIG01_CLASS_COUNTS["e2_irrigated"],
                "e2_rainfed": FIG01_CLASS_COUNTS["e2_rainfed"],
            },
        },
        "development_tag": {
            "id": "e0_tag",
            "label": "E0 · Model-Form Selection",
            "string_class": "direct_label",
            "attached_to": "e2_map",
            "placement": "adjacent_to_e2_heading",
            "placement_detail": (
                "set the tag next to the E2 map heading, so its scope is immediate: E0 ran at "
                "37 of the E2 sites, none of which occur in E1 (handoff sections 6.4 and 7). "
                "The tag must not drift below the map, where it reads as a detached footnote. "
                "A proof may fold the tag into the E2 heading itself at equal weight "
                "(user ruling 2026-08-27 on heading weight)."
            ),
            "forbidden_placements": [
                "beside the E1 heading or the E1 parameter-relay origin",
                "below the E2 map as a detached footnote",
                "in the lower page margin",
                "as a fourth map or another geography",
                "as a coequal evaluation branch",
            ],
            "rendering": (
                "one small, subordinate tag beside the E2 heading, or folded into that heading "
                "at equal weight. E0 is not a fourth map, not a coequal evaluation branch, and "
                "carries no flux-to-parameter arrow."
            ),
        },
        "cartography": {
            "shared_language": [
                "neutral land and restrained boundaries",
                "consistent point halo, opacity and apparent size",
                "comparable label hierarchy",
                "no heavy colored frames",
                "sparse graticules or scale bars only when they improve interpretation",
                "no north arrows on plainly north-up locator maps",
            ],
            "disclaimer_ownership": "caption",
            "context_decision": {
                "decided": "2026-08-25, handoff section 6.7",
                "supersedes": (
                    "the 3.1.0/3.1.1 'context_test' block, which left both archived layers for "
                    "the proof to decide. Revision 3 tested them and the handoff records the "
                    "outcome, so the decision is frozen here."
                ),
                "conus_states": {
                    "layer": "conus_states_context",
                    "status": "ACCEPTED and drawn",
                    "basis": (
                        "'Retain the faint CONUS state boundaries accepted in revision 3' "
                        "(section 6.7)"
                    ),
                    "treatment": (
                        "hairline neutral boundaries at 0.3-0.4 pt and low opacity, strictly "
                        "below the site marks and the coastline in visual weight; no state "
                        "labels, no fills"
                    ),
                },
                "e3_basin": {
                    "layer": "slv_basin_context",
                    "status": "REJECTED; archived but not drawn",
                    "basis": (
                        "'Keep the HUC8 subdivisions out of the E3 map; they competed with the "
                        "field marks and implied an analytical spatial grouping that E3 does not "
                        "use' (section 6.7)"
                    ),
                    "retained_in_package": True,
                    "retention_reason": (
                        "the layer stays in fig01_scope.gpkg as archived, licensed and hashed "
                        "provenance; deleting it would be a Level 3 geography change that alters "
                        "frozen values for no scientific gain. It simply must not be rendered."
                    ),
                    "privacy": (
                        "HUC8 units span 1,987-6,576 km2 and every generalized display point "
                        "falls inside the layer, so publishing it adds no location precision "
                        "beyond the approved 1 km centroid snap"
                    ),
                },
                "e3_boundary_retained": "slv_context, the generalized San Luis Valley boundary",
                "hillshade": (
                    "optional. Section 6.7: 'A subtle privacy-safe hillshade remains optional and "
                    "requires an archived, licensed, hashed source. Its absence does not block "
                    "Gate B.'"
                ),
            },
        },
        "typography": {
            "family": "Source Sans 3, embedded, editable vector text",
            "panel_label_pt": [10, 11],
            "panel_heading_pt": [8.5, 9],
            "structural_label_pt": [8, 8.5],
            "direct_label_and_axis_pt": [7.5, 8],
            "minimum_reader_facing_pt": 7.5,
            "row_label_pt": [7.5, 8],
            "row_label_treatment": (
                "the five panel (a) region names are conventional compact facet or y-axis "
                "labels, set at direct-label weight and size. Their typographic weight must stay "
                "below the panel heading and below the data. They are not five prominent display "
                "headings competing with the marks: architecture 3.0.0 classified them as titles "
                "and the 2026-08-24 revision demoted them to direct labels (handoff section 9)."
            ),
            "case_rule": (
                "title case for structural headings, sentence case for axis text and short "
                "explanatory phrases; conventional acronym capitalization preserved"
            ),
            "identification_color": FIG01_IDENTIFICATION_COLOR,
            "identification_color_rule": (
                "handoff section 9 (2026-08-25): 'Use near-black (#202124 or equivalent) for "
                "reader-facing identification, titles, row names, units, map count lines, "
                "inverse-stage labels, and the US-Bi1 (2017) identifier' (stated for the "
                "then-selected record; the rule applies to the current record identifier "
                f"'{FIG01_RECORD_ID}'). 'Establish hierarchy with size, weight, and placement "
                "-- not by washing secondary information into gray.' Structural cycle labels "
                "stay near-black even when their arrows are purple. The 2026-08-27 style guide "
                "moved reader-facing text to full black #000000, which strictly satisfies the "
                "never-muted-gray intent."
            ),
            "near_black_required_for": [
                "the record identification",
                "panel and map headings",
                "plotting-region names",
                "units",
                "map count lines",
                "inverse-stage labels",
                "the Conditioned Parameters exit label",
            ],
            "muted_gray_scope": [
                "axes",
                "reference rules",
                "quiet geographic context",
                "other nonverbal scaffolding",
            ],
            "muted_gray_forbidden_for": [
                "the record identification",
                "plotting-region names",
                "units",
                "map count lines",
                "inverse-stage labels",
                "any generic secondary-label style",
            ],
            "e0_subordination": (
                "E0 may remain subordinate through smaller type and placement, but it must still "
                "be readily legible at final size; it is not washed into gray (section 9)"
            ),
            "forbidden": [
                "overall title",
                "all-caps headings",
                "paragraph-like annotations",
                "region names set at or above the panel-heading weight",
                "gray text used as a generic secondary-label style",
                "concepts placed inside tinted cards",
            ],
        },
        "color_roles": {
            "satellite_et_target": {
                "hex": "#E69F00",
                "redundant_encoding": "distinct symbol required",
            },
            "swim_state_and_output": {"hex": "#0072B2"},
            "inverse_estimation": {"hex": "#7B3294"},
            "held_out_observation": {"hex": "#202124"},
            "e1": {"hex": "#4477AA"},
            "e2": {"hex": "#228833"},
            "e3": {"hex": "#AA3377"},
            "background": "white",
            "reader_facing_identification": {
                "hex": FIG01_IDENTIFICATION_COLOR,
                "applies_to": (
                    "reader-facing identification, titles, region names, units, map count lines "
                    "and inverse-stage labels (handoff section 9)"
                ),
            },
            "muted_gray": (
                "reserved for axes, reference rules, quiet geographic context and other "
                "nonverbal scaffolding; never a generic secondary-label style for text"
            ),
        },
        "line_weights_pt": {
            "axes_and_reference_rules": [0.5, 0.7],
            "data_strokes": [0.8, 1.2],
            "arrows": "consistent; no single arrow may become the dominant mark",
        },
        "forbidden_visual_treatments": [
            "gradients",
            "drop shadows",
            "glossy icons",
            "thick rounded boxes",
            "decorative satellite art",
            "generic crop artwork or a plant-soil cross-section",
            "a transfer ribbon or process-card matrix",
            "connecting or smoothing raw acquisition points",
            "a zero-valued SWE trace",
            "a data-like meter glyph -- the reviewed proof's three-circle mark included -- while "
            "no E3 observations are frozen",
            "an actual trace compressed into a side strip beside a full-width trace it is "
            "compared with",
            "an actual trace and a symbolic glyph paired inside one apparently equivalent "
            "evidence region",
            "an arrowhead directed into flux or meter observations",
            "a bracket or tie spanning an entire seasonal record",
            "a second irrigation-event encoding alongside the applied-water stems",
            "a card or box around the inverse cycle",
            "a faint driver lane that reads as an axis or y-tick spine",
            "an E3 transfer path routed along or beneath the E2 frame",
            "more than five full-height plotting regions in panel (a)",
            "a plotting region without a visible y-spine, two labelled bound values and units",
            "irrigation stems normalized to the depletion scale instead of carrying their own",
            "the SWIM-RS and flux ET traces separated into independent full-height lanes, or "
            "drawn on different date or vertical mappings",
            "two disconnected L-shaped legs, or a generic one-way transformation, in place of "
            "the closed inverse cycle",
            "the record identification set as muted gray secondary text or as a middle-dot "
            "subtitle",
            "gray text used as a generic secondary-label style anywhere in the figure",
            "the HUC8 subdivisions drawn on the E3 map",
        ],
        "annotation_budget": {
            "maximum": 3,
            "used": 0,
            "panel_a": None,
            "panel_b": None,
            "rule": "seven words or fewer, at most two lines, sentence case, no full sentences",
            "note": (
                "Handoff section 9 targets almost no explanatory annotation. Zero are frozen. "
                "Density comes from real marks, shared axes, direct labels and symbol encodings."
            ),
        },
        "edges": [
            ["etf_ensemble", "compare"],
            ["run_balance", "compare"],
            ["compare", "update_parameters"],
            ["update_parameters", "run_balance"],
            ["update_parameters", "daily_balance"],
            ["ndvi_captures", "run_balance"],
            ["daily_forcing", "run_balance"],
            ["daily_balance", "daily_et"],
            ["daily_balance", "applied_water"],
            ["e3_applied_water", "e3_annual_sum"],
            ["e0_tag", "e1_map"],
            ["e1_map", "irrigated_params"],
            ["e1_map", "rainfed_params"],
            ["irrigated_params", "e2_map"],
            ["irrigated_params", "e3_map"],
            ["rainfed_params", "e2_map"],
        ],
        "edge_labels": {
            "update_parameters -> daily_balance": FIG01_CYCLE_EXIT_LABEL,
        },
        "cycle_edges": [list(e) for e in FIG01_CYCLE_EDGES],
        "cycle_exit_edge": list(FIG01_CYCLE_EXIT_EDGE),
        "comparisons": [
            {
                "a": "daily_et",
                "b": "flux_et",
                "kind": "shared alignment",
                "directed": False,
                "treatment": (
                    "SHARED ALIGNMENT ONLY. In 3.2.0 the two traces are lanes of ONE plotting "
                    "region (handoff section 5.2 item 5), sharing one axis, one date mapping and "
                    "one vertical scale, distinguished by two direct labels. NO bracket, tie "
                    "glyph, or connector of any kind is drawn between them, and no arrowhead may "
                    "point into the flux lane. Section 5.2 also forbids an example metric, a "
                    "residual annotation, a causal arrow, and a separate comparison bracket."
                ),
                "drawn_glyph": None,
                "region": "et_comparison",
                "scope": f"E1 example, {FIG01_EXAMPLE_SITE}",
            },
            {
                "a": "e3_annual_sum",
                "b": "meters",
                "kind": "neutral tie",
                "directed": False,
                "treatment": (
                    "a plain em dash between the two terms inside the separate E3 evaluation "
                    "key, as written in handoff sections 4 and 5.4. No arrowhead into the meter "
                    "term, and no data marks on either side while no E3 observations are frozen."
                ),
                "drawn_glyph": "—",
                "scope": "E3 experiment-level key; not the E1 example record",
            },
        ],
        "comparison_rules": [
            "a comparison is carried by shared alignment or a neutral typographic relation, "
            "never by an arrow that could imply the model produced the observation",
            "the daily ET comparison carries NO drawn glyph at all: one shared axis, one date "
            "mapping, one vertical scale and two direct labels are the entire treatment",
            "the two sides of a comparison must be on an honest common support: one common date "
            "and y mapping for the daily ET pair, one common annual aggregation for the E3 pair",
            "no comparison may join a node of the E1 example to a node of the E3 evaluation key",
        ],
        "forbidden_edges": [
            ["flux_et", "run_balance"],
            ["meters", "run_balance"],
            ["flux_et", "compare"],
            ["meters", "compare"],
            ["flux_et", "update_parameters"],
            ["meters", "update_parameters"],
            ["flux_et", "inverse_estimation"],
            ["meters", "inverse_estimation"],
            ["flux_et", "daily_balance"],
            ["meters", "daily_balance"],
            ["flux_et", "irrigated_params"],
            ["meters", "irrigated_params"],
            ["flux_et", "rainfed_params"],
            ["meters", "rainfed_params"],
            ["flux_et", "e1_map"],
            ["meters", "e1_map"],
            ["flux_et", "e2_map"],
            ["meters", "e3_map"],
            ["e2_map", "e3_map"],
            ["e3_map", "e2_map"],
            ["e2_map", "irrigated_params"],
            ["e2_map", "rainfed_params"],
            ["rainfed_params", "e3_map"],
            ["applied_water", "meters"],
            ["applied_water", "e3_annual_sum"],
            ["applied_water", "e3_applied_water"],
            ["e3_annual_sum", "applied_water"],
            ["e3_annual_sum", "meters"],
            ["daily_et", "flux_et"],
            ["swe_constraint", "compare"],
            ["swe_inline", "compare"],
            ["swe_constraint", "inverse_estimation"],
            ["swe_inline", "inverse_estimation"],
            ["etf_ensemble", "run_balance"],
            ["ndvi_captures", "compare"],
            ["daily_forcing", "compare"],
        ],
        "forbidden_edge_reasons": {
            "applied_water -> meters": (
                "the linkage architecture 3.0.0 asserted and this revision removes. The E1 "
                "example's simulated irrigation is not an E3 meter pair; E3 evaluates applied "
                "water simulated for E3 fields."
            ),
            "applied_water -> e3_annual_sum / e3_applied_water": (
                "the same false linkage routed through the E3 key's own nodes"
            ),
            "e3_annual_sum -> applied_water": "the same linkage in reverse",
            "e3_annual_sum -> meters": (
                "the E3 relation is a neutral comparison tie, not an arrow into a held-out "
                "observation"
            ),
            "daily_et -> flux_et": (
                "the two traces share one axis; their relationship is alignment, not an arrow "
                "implying the model produced the observation"
            ),
            "flux_et / meters -> run_balance, compare, update_parameters": (
                "handoff section 10.3: 'Held-out flux and meter observations remain forbidden as "
                "cycle inputs.' Every stage of the closed cycle is protected, not just the "
                "retired single inverse node."
            ),
            "swe_constraint / swe_inline -> compare or inverse_estimation": (
                "SWE is an inline label on the incoming ETf constraint route, not a separate "
                "node with its own arrow"
            ),
            "etf_ensemble -> run_balance": (
                "the acquisition-date targets enter the COMPARE stage. Routing them into the "
                "balance would say the satellite retrievals drive the forward model rather than "
                "constrain its parameters (handoff section 5.3)."
            ),
            "ndvi_captures / daily_forcing -> compare": (
                "NDVI and daily forcing drive the forward balance; they are not calibration "
                "targets and must not enter the comparison stage (handoff section 5.3)"
            ),
        },
        "edge_rules": [
            "no arrow may run from a held-out observation into any stage of the inverse cycle, "
            "the displayed daily balance, or either class-specific parameter set",
            "no edge may originate at a held-out observation at all",
            "comparison with a held-out observation is a neutral tie recorded under comparisons, "
            "never a directed edge; no arrowhead points into flux or meter observations",
            "no drawn relationship of any kind -- edge, comparison, bracket, tie or shared glyph "
            "-- may connect the E1 example's applied-water node to the E3 meter node, directly "
            "or through any intermediate node",
            "both transfer paths originate at the E1 map; there is no E2-to-E3 edge",
            "E2 receives both parameter classes; E3 receives only the irrigated class",
            "the E3 path branches visibly at the irrigated parameter token, before the E2 frame",
            "the ETf constraint path and the NDVI/forcing driver path must be visually "
            "distinguishable by routing and stroke treatment; SWE rides the constraint path as "
            "an inline label rather than its own arrow",
            "the constraint route enters 'Compare'; the driver route enters 'Run Balance'; the "
            "three cycle edges close the loop in the order Run Balance -> Compare -> Update "
            "Parameters -> Run Balance",
            "the 'Conditioned Parameters' exit leaves the update stage for the displayed daily "
            "trajectory and is the only edge out of the cycle",
            "panels (a) and (b) are not joined by an arrow; the removed inverse_estimation -> "
            "e1_source connector belonged to the superseded transfer-ribbon composition",
        ],
        "open_decisions": [
            "No E3 meter observation values are frozen in the Figure 1 package, so the E3 "
            "evaluation key stays typographic and schematic. Drawing real meter marks requires "
            "freezing observations chosen by a support rule independent of simulated "
            "performance, the corresponding E3 model output on an honest common support, and a "
            "separate selection record (handoff section 5.4).",
            "'PEST++ IES' is removed from the artwork and carried by the caption. Handoff "
            "section 5.3 permits its return only if a final-size proof shows it adds useful "
            "specificity without enlarging the inverse element; that proof has not been made.",
            "'spread-weighted' is removed from the artwork and carried by the caption. Handoff "
            "section 5.3 permits its return on the incoming target route only if a final-size "
            "proof shows it materially clarifies the member-spread encoding; that proof has not "
            "been made, and a return would also require restoring its caption-guard exemption.",
            "The 'daily drivers' bracket is conditional: handoff section 5.3 says 'if retained' "
            "it must span only the NDVI and forcing regions and use one short connector into "
            "Run Balance. Whether it survives at final size is a proof-level decision; it stays "
            "in the frozen copy so a proof that keeps it draws the agreed string.",
            "A faint privacy-safe E3 hillshade is optional. Handoff section 6.7: 'A subtle "
            "privacy-safe hillshade remains optional and requires an archived, licensed, hashed "
            "source. Its absence does not block Gate B.' It may only be attempted once a "
            "licensed DEM and its derivation are archived and hashed the way "
            "conus_states_context and slv_basin_context are, and it must not resolve terrain "
            "finely enough to localize a field beyond the approved 1 km centroid snap.",
        ],
        "decisions_closed_2026_08_25": [
            "The closed inverse cycle, carried as an open decision since 3.0.0, is now REQUIRED. "
            "Handoff sections 5.3, 10.3 and 11 mandate the directed cycle 'Run Balance -> "
            "Compare -> Update Parameters -> Run Balance' and the 'Conditioned Parameters' exit; "
            "the single one-way parameter-update arrow is superseded.",
            "conus_states_context is ACCEPTED and drawn: 'Retain the faint CONUS state "
            "boundaries accepted in revision 3' (section 6.7).",
            "slv_basin_context is REJECTED as artwork: 'Keep the HUC8 subdivisions out of the E3 "
            "map' (section 6.7). The layer stays in fig01_scope.gpkg as archived provenance and "
            "is simply not rendered.",
            "Panel (b) is ACCEPTED as built in revision 3 (section 6.1); subsequent changes to "
            "arc curvature, map positions, clearances, or stroke weights are render-only unless "
            "they change the source or destination of a path.",
        ],
        "panel_b_status": {
            "revision": "r3",
            "accepted": True,
            "source": "handoff section 6.1",
            "accepted_topology": (
                "the irrigated triangle is the fork, one horizontal leg reaches E2, and one "
                "visibly divergent curved leg reaches E3"
            ),
            "subsequent_changes": (
                "arc curvature, map positions, clearances and stroke weights are render-only "
                "(Level 1) adjustments unless they change the source or destination of a path, "
                "which would be a Level 2 contract change"
            ),
            "redesign_scope_2026_08_25": "panel (a) only",
        },
        "revision_protocol": {
            "source": "handoff section 15, 'Fast revision and rebuild protocol'",
            "rule": (
                "classify every requested change before editing and use the lowest level that "
                "preserves scientific meaning and provenance. Contract review, background agents "
                "and 'build_figure_data.py --all' are not gates for geometry or style work."
            ),
            "levels": {
                "0_svg_markup": (
                    "a reviewer moves, resizes or restyles existing elements in Inkscape to "
                    "communicate a preference; edit a copy named '*_markup.svg' only, rebuild "
                    "nothing"
                ),
                "1_render_only": (
                    "positions, margins, panel heights, axis limits, generated tick placement, "
                    "font sizes or weights, colors, line weights, map extents using the same "
                    "geometries, connector routing; active proof script only, no figure-data "
                    "build"
                ),
                "2_contract": (
                    "any visible string; adding or removing a label; node, edge, comparison or "
                    "held-out semantics; panel grouping; data role; or relationship change. "
                    "Rebuild through 'uv run python scripts/figures/build_figure_data.py --all' "
                    "and regenerate the architecture, metadata and manifest records."
                ),
                "3_display_data": (
                    "new or changed plotted values, columns, transformations, site/window "
                    "selection, aggregation, cohort/class assignment, geography, or privacy "
                    "generalization; full '--all' build plus a full scientific, provenance, "
                    "selection, cohort, privacy and render audit"
                ),
            },
            "this_change": (
                "Level 2. Handoff section 16: 'This 2026-08-25 redesign is a Level 2 contract "
                "change because it changes visible copy, panel grouping, and "
                "inverse-estimation relationships.'"
            ),
            "level_2_value_equality_required_for": [
                "fig01_example_timeseries.csv",
                "fig01_example_selection.json",
                "fig01_scope.gpkg",
                "fig01_evidence_matrix.csv",
            ],
            "level_2_value_equality_rule": (
                "'A changed value in those files without a declared Level 3 reason is a failure.' "
                "fig01_architecture.json, contract-bearing metadata and manifest entries are the "
                "expected changes. fig01_scope.gpkg differs by its gpkg_contents.last_change "
                "timestamp on every build, so layer content -- not the file hash -- is the valid "
                "equality test."
            ),
            "axis_domain_note": (
                "axis-domain and tick changes stay Level 1 when they use the same frozen "
                "columns, do not transform values, and do not clip marks"
            ),
            "svg_markup_channel": {
                "source": "handoff section 15.2",
                "requirement": (
                    "every proof builder must write an editable SVG alongside the PDF, PNG, "
                    "string ledger and notes, on the same 190 x 120 mm canvas, assigning stable "
                    "IDs to panels, axes, labels, routes, map groups and data-mark groups so "
                    "markup deltas can be identified reliably"
                ),
                "cycle": [
                    "render the deterministic script to SVG/PDF/PNG",
                    "copy the SVG to '*_markup.svg' and edit that copy in Inkscape",
                    "read the moved elements and transforms from the markup copy",
                    "transcribe accepted deltas into named constants or layout logic in the "
                    "proof script",
                    "rerender cleanly and compare the generated SVG with the markup intent",
                ],
                "rule": (
                    "the markup SVG is a communication artifact, not a publication artifact and "
                    "not the source of record. Never use it to hand-edit data positions, visible "
                    "copy, or scientific relationships; only clean scripted outputs may advance "
                    "to Gate B."
                ),
                "stable_id_groups": [
                    "panels",
                    "axes",
                    "labels",
                    "routes",
                    "map groups",
                    "data-mark groups",
                ],
            },
            "proof_directory_practice": (
                "handoff section 15.3: preserve revisions 1-3 as review provenance, create one "
                "active revision-4 working directory, and start a new numbered directory after a "
                "Level 2 or Level 3 change. Architecture 3.2.0 IS a Level 2 change, so revision "
                "4 must be written to a new proof directory; fig01_graybox_110.png, "
                "fig01_evidence_190.png and every accepted r2/r3 artifact stay untouched."
            ),
        },
        "superseded_requirements_removed": {
            "source": "handoff section 10.3",
            "instruction": (
                "'Remove superseded seven-lane, one-way inverse, transfer-ribbon, and "
                "crop-cross-section requirements.'"
            ),
            "seven_lane": (
                "REMOVED. Replaced by the five plotting_regions; the evidence_rows key itself is "
                "gone so no downstream script can read the old grouping."
            ),
            "one_way_inverse": (
                "REMOVED. Replaced by inverse_cycle with the three cycle edges and the labelled "
                "Conditioned Parameters exit; the etf_ensemble -> inverse_estimation and "
                "inverse_estimation -> daily_balance edges are gone from the edge list."
            ),
            "transfer_ribbon": (
                "REMOVED in 3.0.0 and kept out. The unlettered full-width ribbon and its three "
                "ribbon nodes remain in also_supersedes (2.1.0) and in "
                "forbidden_visual_treatments; the maps themselves are the transfer endpoints "
                "(handoff section 4)."
            ),
            "crop_cross_section": (
                "REMOVED in 3.0.0 and kept out. The generic crop-soil-water process "
                "cross-section stays in forbidden_visual_treatments; handoff section 5.2 item 4 "
                "replaces it with the coordinated state-and-irrigation region."
            ),
        },
        "string_classification": {},
        "caption_facts": {
            "_ownership": "caption/manuscript-owned",
            "_rule": (
                "Recorded here for caption drafting and audit only. The builder never promotes "
                "any of these strings into visible copy, and an assertion fails the build if one "
                "appears among strings classified title, direct_label, or annotation."
            ),
            "contract": (
                "paper/notes/fig01_production_handoff.md section 12, revised 2026-08-24, eight "
                "required items"
            ),
            "item_1_example_independence": (
                "Panel (a) shows one illustrative E1 growing-season record, selected without "
                "reference to evaluation performance; it supports no performance claim."
            ),
            "item_2_sparse_versus_daily": (
                "Acquisition-date satellite ETf members are calibration targets at 15 dates; raw "
                "Landsat and Sentinel-2 NDVI and daily meteorological forcing drive the forward "
                "balance, which carries soil-water state between acquisitions."
            ),
            "item_3_spread_weighting": (
                "Retrieval-member dispersion sets the relative weight of each acquisition-date "
                "target: captures on which the members disagree constrain the parameters less. "
                "Inverse estimation uses PEST++ IES. This is relative reliability information, "
                "not calibrated uncertainty. The objective and the weight expression stay in the "
                "Methods."
            ),
            "item_4_outputs": (
                "The E1 example shows two kinds of daily evidence: the model's daily ET, aligned "
                "with the held-out flux record on a common date axis, and its simulated "
                "irrigation events, shown as magnitude-bearing applied-water stems that "
                "illustrate the model's daily output form. Separately, and not as an extension "
                "of that record, E3 aggregates daily gross applied water from E3 simulations to "
                "annual totals for comparison with meter records. The plotted E1 applied-water "
                "series is not an E3 meter pair."
            ),
            "item_4_note": (
                "Revised 2026-08-24. Architecture 3.0.0's caption conflated the two: it read the "
                "E1 example's applied water as the series aggregated for E3, which the "
                "artwork then drew as a linkage. Handoff section 12 item 4 now requires the "
                "distinction."
            ),
            "item_5_held_out": (
                "Flux ET evaluates E1 and E2; metered applied water evaluates E3. Flux ET and "
                "meter records were withheld from parameter estimation and from transfer-vector "
                "construction in every experiment. Where a model output and an observation are "
                "drawn together they are aligned for post-hoc comparison only; no arrow runs "
                "from a model output into an observation."
            ),
            "item_5_note": (
                "3.1.1 (review directive d1): the flux lane in panel (a) is now labelled simply "
                "'Flux ET' because the drawn trace is one single-site E1 example record. The E1-E2 "
                "flux-evaluation scope that the 'Flux ET · E1–E2' label used to assert is "
                "caption-owned and is carried by the first sentence of this item. The sentence "
                "about arrows was also corrected: since 3.1.0 there is no model-output-to-"
                "observation arrow to qualify, and since 3.1.1 the daily ET comparison carries "
                "no drawn glyph at all."
            ),
            "item_6_scope_and_classes": (
                "Configured scope is 60 CONUS cropland sites (E1), 66 cropland sites in ten "
                "countries on four continents (E2), and 50 metered San Luis Valley fields (E3). "
                "E1 supplies two class-specific parameter sets: irrigated (39 source sites) and "
                "rainfed (21 source sites). Both are applied to E2, which is assigned 13 "
                "irrigated and 53 rainfed; only the irrigated set is applied to E3. E2 comprises "
                "13 sites shared with E1 under changed inputs plus 53 new fields."
            ),
            "item_7_e0_disclosure": (
                "E0 compared three vegetation formulations at 37 of the E2 sites, none of which "
                "occur in E1, using the E2 inputs and processing. Flux ET did not enter "
                "parameter estimation; it was used after satellite calibration to compare the "
                "formulations and select the cover-scaled sigmoid carried into E1-E3. E0 is "
                "model-development evidence, not independent validation, and not a fourth "
                "geography."
            ),
            "item_8_map_disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted national "
                "boundaries."
            ),
            "working_caption": (
                "Fig. 1. Observation, state-propagation, evaluation, and transfer design of "
                "SWIM-RS. (a) An illustrative E1 growing-season record, selected without "
                "reference to evaluation performance, shows the temporal information entering "
                "and leaving the framework. Raw Landsat and Sentinel-2 NDVI observations "
                "represent vegetation dynamics; acquisition-date satellite ETf members and "
                "gridded SWE constrain time-invariant parameters through spread-weighted inverse "
                "estimation with PEST++ IES; and daily meteorological forcing drives a "
                "mass-conserving balance that carries soil-water state between acquisitions. The "
                "example aligns daily SWIM-RS ET with held-out flux ET on a common date axis and "
                "shows simulated irrigation events as evidence of the model's daily output form. "
                "Separately, E3 aggregates daily gross applied water from E3 simulations to "
                "annual totals for comparison with meter records; the plotted E1 applied-water "
                "record is not an E3 meter pair. Flux and meter observations were withheld from "
                "parameter estimation and transferred-parameter construction. (b) The 60 E1 "
                "CONUS sites supply separate irrigated and rainfed parameter sets. Both are "
                "applied without field-specific calibration across the 66-site, ten-country E2 "
                "experiment, whereas the irrigated set is applied to 50 metered fields in the "
                "San Luis Valley. E0 used 37 of the E2 sites, none of which occur in E1, to "
                "compare three vegetation formulations against flux ET after satellite "
                "calibration and select the formulation carried into E1-E3. Map lines "
                "delineate study areas and do not necessarily depict accepted national "
                "boundaries."
            ),
            "working_caption_source": (
                "paper/notes/fig01_production_handoff.md section 12, revised 2026-08-24, "
                "transcribed verbatim"
            ),
            "working_caption_status": (
                "to be reconciled with the finished render (handoff section 12 and section 15 "
                "item 3)"
            ),
            "caption_owned_engine_label": (
                "PEST++ IES. Removed from the artwork by the 2026-08-24 revision (handoff "
                "section 5.3) and carried here."
            ),
            "objective_notation": "Φ = Σ_i [w_i(ETf_sim,i − ETf_obs,i)]² + Σ_j [w_j(SWE_sim,j − SWE_obs,j)]²",
            "weighting_treatment": {
                "primary_experiments": "spread-weighted (all of E1, E2, E3 primaries)",
                "weight_rule": "w_i = ETf_obs,i / (σ_ensemble,i + 0.1); relative reliability information, not calibrated uncertainty",
                "fixed_scale_exception": "a fixed 0.33 scale is used only for the E2 ECOSTRESS-only sensitivity rows",
                "localization": "SWE observations update only the two snow parameters",
            },
            "retrieval_members": {
                "openet_v21_members": E1_MEMBERS,
                "E0": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf, matched across the three candidate formulations",
                "E1": "per-capture mean of the six OpenET v2.1 ETf members",
                "E2": "per-capture mean of coincident Landsat SSEBop and PT-JPL ETf",
                "E2_sensitivity": "ECOSTRESS ETf on Landsat-gap dates at a fixed 0.33 scale",
                "E3": "per-capture mean of the six OpenET v2.1 ETf members, at least two valid members",
            },
            "calibration_settings": {
                "engine": "PEST++ IES",
                "prior_realizations": 200,
                "iterations": "three or four, by experiment",
                "n_parameters": 8,
            },
            "paired_support": {
                "E0": "37 daily sites; 33 finite-metric monthly sites (37 with paired months)",
                "E1": "45 daily sites; 29 finite-metric monthly sites (31 supported)",
                "E2": "63 daily sites; 50 finite-metric monthly sites (56 supported)",
                "E3": f"{EXPECTED['E3_field_years']} metered field-years across 50 fields",
                "note": "configured scope (60 / 66 / 50) is distinct from paired evaluation support",
            },
            "example_site": (
                f"{FIG01_EXAMPLE_SITE}, {FIG01_EXAMPLE_START} to {FIG01_EXAMPLE_END}; "
                f"{FIG01_EXAMPLE_DAYS} days and {FIG01_EXAMPLE_CAPTURES} calibration captures"
            ),
            "ecostress_sensitivity": (
                "The E2 ECOSTRESS-only ablation adds capture dates on Landsat gaps at a fixed "
                "0.33 scale. The ECOSTRESS-to-Landsat ETf ratio is 0.756; the daily result is a "
                "wash and the monthly medians sit near the noise floor, so the ablation is "
                "reported as a sensitivity, not as the canonical target."
            ),
            "firewall_qualification": (
                "Flux ET evaluates E1 and E2; metered applied water evaluates E3. Both were "
                "withheld from parameter estimation and from transfer-vector construction in "
                "every experiment. Where a model output and an observation are drawn together "
                "they are aligned for post-hoc comparison only; no arrow runs from a model "
                "output into an observation."
            ),
            "e0_qualification": (
                "E0 is a vegetation-formulation comparison at 37 of the E2 sites, none of which "
                "occur in E1, run on the E2 inputs and processing. Flux ET did not enter "
                "parameter estimation, but E0 used flux ET after satellite calibration to "
                "select the cover-scaled sigmoid formulation carried by E1-E3. E0 is "
                "model-development evidence, not independent validation, and is not a fourth "
                "geography."
            ),
            "transfer_definition": (
                "The panel (b) tokens denote fixed E1-derived irrigation-class parameter sets: "
                "the irrigated and rainfed sets are applied to E2, and the irrigated set to E3. "
                "E2 does not supply parameters to E3."
            ),
            "map_disclaimer": (
                "Map lines delineate study areas and do not necessarily depict accepted national "
                "boundaries."
            ),
        },
    }
