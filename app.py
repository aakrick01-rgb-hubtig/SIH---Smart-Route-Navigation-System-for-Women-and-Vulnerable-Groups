from pathlib import Path
import csv

import streamlit as st
import networkx as nx
import folium

from streamlit_folium import st_folium


# ============================================================
# SAFEPath AI
# ML SAFETY GRAPH + ORIGINAL STABLE FRONTEND
# ============================================================

st.set_page_config(
    page_title="SafePath AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "sos_active" not in st.session_state:
    st.session_state["sos_active"] = False

if "sos_destination" not in st.session_state:
    st.session_state["sos_destination"] = None


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

GRAPH_PATH = (
    BASE
    / "data"
    / "processed"
    / "safepath_ml_safety_graph.graphml"
)

AMENITIES_PATH = (
    BASE
    / "data"
    / "processed"
    / "amenities_clean.csv"
)

MAX_CANDIDATES = 500


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    .hero {
        padding: 1.1rem 1.35rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        margin: .35rem 0 0;
        color: #cbd5e1;
    }

    .small-note {
        font-size: .85rem;
        color: #64748b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ SafePath AI</h1>
        <p>
            Safety-aware urban navigation using ML-predicted
            road safety, spatial risk analysis and intelligent routing.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ML GRAPH
# ============================================================

@st.cache_resource
def load_ml_graph():

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            f"ML safety graph not found:\n{GRAPH_PATH}"
        )

    G = nx.read_graphml(
        GRAPH_PATH
    )

    # --------------------------------------------------------
    # Validate node coordinates
    # --------------------------------------------------------

    invalid_nodes = []

    for node, data in G.nodes(
        data=True
    ):

        try:

            if (
                "x" not in data
                or "y" not in data
            ):

                invalid_nodes.append(node)
                continue

            data["x"] = float(
                data["x"]
            )

            data["y"] = float(
                data["y"]
            )

        except (
            ValueError,
            TypeError
        ):

            invalid_nodes.append(node)


    if invalid_nodes:

        G.remove_nodes_from(
            invalid_nodes
        )


    # --------------------------------------------------------
    # Clean edge attributes
    # --------------------------------------------------------

    for u, v, data in G.edges(
        data=True
    ):

        # Length
        try:

            data["length"] = float(
                data.get(
                    "length",
                    1.0
                )
            )

            if data["length"] <= 0:
                data["length"] = 1.0

        except (
            ValueError,
            TypeError
        ):

            data["length"] = 1.0


        # Original safety score
        try:

            data["safety_score"] = float(
                data.get(
                    "safety_score",
                    50.0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            data["safety_score"] = 50.0


        # ML predicted safety
        try:

            data["predicted_safety_score"] = float(
                data.get(
                    "predicted_safety_score",
                    data["safety_score"]
                )
            )

        except (
            ValueError,
            TypeError
        ):

            data["predicted_safety_score"] = (
                data["safety_score"]
            )


        # Baseline score
        try:

            data["baseline_safety_score"] = float(
                data.get(
                    "baseline_safety_score",
                    data["safety_score"]
                )
            )

        except (
            ValueError,
            TypeError
        ):

            data["baseline_safety_score"] = (
                data["safety_score"]
            )


        # Safety risk
        try:

            data["safety_risk"] = float(
                data.get(
                    "safety_risk",
                    100.0
                    - data["predicted_safety_score"]
                )
            )

        except (
            ValueError,
            TypeError
        ):

            data["safety_risk"] = (
                100.0
                - data["predicted_safety_score"]
            )


        # Clamp values

        data["safety_score"] = max(
            0.0,
            min(
                100.0,
                data["safety_score"]
            )
        )

        data["predicted_safety_score"] = max(
            0.0,
            min(
                100.0,
                data["predicted_safety_score"]
            )
        )

        data["safety_risk"] = max(
            0.0,
            data["safety_risk"]
        )


    if len(G.nodes) < 2:

        raise RuntimeError(
            "The ML safety graph needs at least two "
            "usable nodes."
        )


    if len(G.edges) == 0:

        raise RuntimeError(
            "The ML safety graph contains no road edges."
        )


    return G


# ============================================================
# CREATE ROUTING GRAPH
# ============================================================

def create_routing_graph(G):

    U = nx.Graph()


    # --------------------------------------------------------
    # IMPORTANT:
    # Copy ALL node attributes including x/y.
    # --------------------------------------------------------

    U.add_nodes_from(
        G.nodes(
            data=True
        )
    )


    # --------------------------------------------------------
    # Convert directed / multigraph to simple graph
    # --------------------------------------------------------

    try:

        undirected = G.to_undirected()

    except Exception:

        undirected = G


    for u, v, data in undirected.edges(
        data=True
    ):

        if (
            u not in U.nodes
            or v not in U.nodes
        ):

            continue


        # Length

        try:

            length = float(
                data.get(
                    "length",
                    1.0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            length = 1.0


        if length <= 0:

            length = 1.0


        edge_data = dict(
            data
        )

        edge_data["length"] = length


        # ML safety score

        try:

            edge_data["predicted_safety_score"] = float(
                edge_data.get(
                    "predicted_safety_score",
                    edge_data.get(
                        "safety_score",
                        50.0
                    )
                )
            )

        except (
            ValueError,
            TypeError
        ):

            edge_data["predicted_safety_score"] = 50.0


        # Safety score

        try:

            edge_data["safety_score"] = float(
                edge_data.get(
                    "safety_score",
                    edge_data["predicted_safety_score"]
                )
            )

        except (
            ValueError,
            TypeError
        ):

            edge_data["safety_score"] = (
                edge_data["predicted_safety_score"]
            )


        # Safety risk

        try:

            edge_data["safety_risk"] = float(
                edge_data.get(
                    "safety_risk",
                    max(
                        0.0,
                        100.0
                        - edge_data[
                            "predicted_safety_score"
                        ]
                    )
                )
            )

        except (
            ValueError,
            TypeError
        ):

            edge_data["safety_risk"] = max(
                0.0,
                100.0
                - edge_data[
                    "predicted_safety_score"
                ]
            )


        # ----------------------------------------------------
        # Keep shortest parallel edge
        # ----------------------------------------------------

        if not U.has_edge(
            u,
            v
        ):

            U.add_edge(
                u,
                v,
                **edge_data
            )

        else:

            existing_length = float(
                U[u][v].get(
                    "length",
                    1.0
                )
            )

            if length < existing_length:

                U[u][v].update(
                    edge_data
                )


    # Remove isolated nodes

    U.remove_nodes_from(
        list(
            nx.isolates(U)
        )
    )


    return U


# ============================================================
# LOAD GRAPH
# ============================================================

@st.cache_resource
def get_graphs():

    G = load_ml_graph()

    U = create_routing_graph(
        G
    )

    return G, U


try:

    with st.spinner(
        "Loading ML safety graph..."
    ):

        G, U = get_graphs()

except Exception as exc:

    st.error(
        "SafePath could not load its ML safety graph."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# BUILD HUMAN-READABLE NODE LABELS
# ============================================================

node_items = []

for node, data in U.nodes(
    data=True
):

    try:

        lat = float(
            data["y"]
        )

        lon = float(
            data["x"]
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ):

        continue

    node_items.append(
        (
            node,
            lat,
            lon
        )
    )


if len(node_items) < 2:

    st.error(
        "The safety graph needs at least two usable locations."
    )

    st.stop()


center_lat = (
    sum(
        x[1]
        for x in node_items
    )
    / len(node_items)
)

center_lon = (
    sum(
        x[2]
        for x in node_items
    )
    / len(node_items)
)


def position_label(
    lat,
    lon
):

    ns = (
        "North"
        if lat >= center_lat
        else "South"
    )

    ew = (
        "East"
        if lon >= center_lon
        else "West"
    )

    return (
        f"{ns} {ew} Network Point"
    )


# ============================================================
# HUMAN-FRIENDLY NODE LABELS
# ============================================================

label_by_node = {}

seen = {}


for node, lat, lon in sorted(
    node_items,
    key=lambda x: (
        x[1],
        x[2]
    )
):

    base = position_label(
        lat,
        lon
    )

    seen[base] = (
        seen.get(
            base,
            0
        )
        + 1
    )

    label_by_node[node] = (
        f"{base} {seen[base]}"
    )


def node_label(node):

    return label_by_node.get(
        node,
        str(node)
    )


nodes = [
    x[0]
    for x in node_items
]


# ============================================================
# FIND FAR APART DEFAULT START / END
# ============================================================

best_pair = None

# Use the largest connected component to avoid
# wasting time on unreachable pairs.

components = list(
    nx.connected_components(U)
)

if not components:

    st.error(
        "No connected road network found."
    )

    st.stop()


largest_component = max(
    components,
    key=len
)

component_graph = U.subgraph(
    largest_component
)


# Choose a well-connected start node

farthest_start = max(
    largest_component,
    key=lambda n: component_graph.degree(n)
)


# Find farthest reachable destination

try:

    distances = nx.single_source_dijkstra_path_length(
        component_graph,
        farthest_start,
        weight="length"
    )

    farthest_end = max(
        distances,
        key=distances.get
    )

except Exception:

    farthest_end = next(
        n
        for n in nodes
        if n != farthest_start
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🕹️ SafePath Controls"
)

st.sidebar.markdown(
    "### Route Selection"
)


# ------------------------------------------------------------
# START LOCATION
# ------------------------------------------------------------

start_order = (
    [farthest_start]
    + [
        n
        for n in nodes
        if n != farthest_start
    ]
)


start_node = st.sidebar.selectbox(
    "📍 Start location",
    start_order,
    index=0,
    format_func=node_label
)


# ------------------------------------------------------------
# DESTINATION
# ------------------------------------------------------------

destination_options = [
    n
    for n in nodes
    if n != start_node
]


try:

    distances_from_start = (
        nx.single_source_dijkstra_path_length(
            U,
            start_node,
            weight="length"
        )
    )

except Exception:

    distances_from_start = {}


destination_options.sort(
    key=lambda n: distances_from_start.get(
        n,
        -1
    ),
    reverse=True
)


if not destination_options:

    st.error(
        "No valid destination is reachable from "
        "the selected start."
    )

    st.stop()


# Prefer farthest reachable destination

selected_end_node = st.sidebar.selectbox(
    "🏁 Destination",
    destination_options,
    index=0,
    format_func=node_label
)


# ------------------------------------------------------------
# DETOUR SLIDER
# ------------------------------------------------------------

max_detour = st.sidebar.slider(
    "Maximum acceptable detour",
    min_value=0,
    max_value=25,
    value=10,
    step=1,
    format="%d%%"
)

st.sidebar.caption(
    "Lower detour = prioritize shorter travel. "
    "Higher detour = allow a longer safer alternative."
)


st.sidebar.markdown("---")


# ============================================================
# EMERGENCY MODE
# ============================================================

st.sidebar.subheader(
    "🚨 Emergency Mode"
)


sos_triggered = st.sidebar.button(
    "🚨 ACTIVATE SOS",
    use_container_width=True
)


find_route_triggered = st.sidebar.button(
    "🛡️ FIND SAFEST ROUTE",
    type="primary",
    use_container_width=True
)


# ============================================================
# NORMAL ROUTE BUTTON
# ============================================================

if find_route_triggered:

    st.session_state["sos_active"] = False

    st.session_state[
        "sos_destination"
    ] = None


# ============================================================
# FIND NEAREST NODE
# ============================================================

def nearest_node(
    graph,
    latitude,
    longitude
):

    best_node = None

    best_distance = float(
        "inf"
    )


    for node, data in graph.nodes(
        data=True
    ):

        try:

            node_lat = float(
                data["y"]
            )

            node_lon = float(
                data["x"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue


        distance = (
            (node_lat - latitude) ** 2
            +
            (node_lon - longitude) ** 2
        )


        if distance < best_distance:

            best_distance = distance

            best_node = node


    return best_node


# ============================================================
# SOS ACTIVATION
# ============================================================

if sos_triggered:

    emergency_candidates = []


    # --------------------------------------------------------
    # Read amenities_clean.csv
    # --------------------------------------------------------

    if AMENITIES_PATH.exists():

        emergency_priorities = {

            "Police Station": 0,

            "Police Chowky": 0,

            "Police Checkpoint": 0,

            "Emergency Clinic": 1,

            "Municipal Hospital": 1,

            "Government Hospital": 1,

            "24-7 Pharmacy": 2,

            "Hospital": 1,

            "Clinic": 1

        }


        try:

            with AMENITIES_PATH.open(
                encoding="utf-8-sig",
                newline=""
            ) as f:

                reader = csv.DictReader(
                    f
                )


                for row in reader:

                    kind = str(
                        row.get(
                            "amenity_type",
                            ""
                        )
                    ).strip()


                    if kind not in emergency_priorities:

                        continue


                    verified = str(
                        row.get(
                            "is_verified",
                            ""
                        )
                    ).strip().lower() in {
                        "true",
                        "1",
                        "yes"
                    }


                    if not verified:

                        continue


                    try:

                        latitude = float(
                            row["latitude"]
                        )

                        longitude = float(
                            row["longitude"]
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError
                    ):

                        continue


                    emergency_node = nearest_node(
                        U,
                        latitude,
                        longitude
                    )


                    if emergency_node is None:

                        continue


                    # Never route SOS from a node to itself

                    if emergency_node == start_node:

                        continue


                    # Check reachability

                    try:

                        route_distance = (
                            nx.shortest_path_length(
                                U,
                                start_node,
                                emergency_node,
                                weight="length"
                            )
                        )

                    except nx.NetworkXNoPath:

                        continue


                    emergency_candidates.append(
                        (
                            emergency_node,
                            kind,
                            row.get(
                                "amenity_id",
                                "Emergency Point"
                            ),
                            route_distance,
                            emergency_priorities[
                                kind
                            ]
                        )
                    )


        except Exception:

            emergency_candidates = []


    # --------------------------------------------------------
    # Select best emergency facility
    # --------------------------------------------------------

    if emergency_candidates:

        emergency_candidates.sort(
            key=lambda x: (
                x[4],
                x[3]
            )
        )


        chosen = (
            emergency_candidates[0]
        )


        st.session_state[
            "sos_active"
        ] = True


        st.session_state[
            "sos_destination"
        ] = (
            chosen[0],
            chosen[1],
            chosen[2]
        )


    else:

        st.session_state[
            "sos_active"
        ] = False


        st.session_state[
            "sos_destination"
        ] = None


# ============================================================
# CURRENT SOS STATE
# ============================================================

sos_active = st.session_state.get(
    "sos_active",
    False
)

sos_destination = st.session_state.get(
    "sos_destination"
)


# ============================================================
# DETERMINE FINAL DESTINATION
# ============================================================

end_node = selected_end_node


if (
    sos_active
    and sos_destination
):

    end_node = sos_destination[0]


# ------------------------------------------------------------
# Protection against start == destination
# ------------------------------------------------------------

if end_node == start_node:

    alternatives = [
        n
        for n in destination_options
        if n != start_node
    ]


    if alternatives:

        end_node = alternatives[0]

    else:

        st.error(
            "Please select a different destination."
        )

        st.stop()


# ============================================================
# SOS SIDEBAR STATUS
# ============================================================

if (
    sos_active
    and sos_destination
):

    _, kind, aid = sos_destination


    st.sidebar.success(
        f"🚨 SOS target: {kind}"
    )


    st.sidebar.caption(
        f"Emergency facility: {aid}"
    )


    st.sidebar.caption(
        "Routing to a verified emergency facility "
        "reachable from the selected start."
    )


elif sos_triggered:

    st.sidebar.warning(
        "No verified emergency facility was found "
        "in the current network."
    )


# ============================================================
# ROUTE STATISTICS
# ============================================================

def route_stats(
    graph,
    route
):

    distance = 0.0

    total_risk = 0.0

    scores = []


    if (
        route is None
        or len(route) < 2
    ):

        return (
            0.0,
            0.0,
            0.0
        )


    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        if not graph.has_edge(
            u,
            v
        ):

            continue


        data = graph[u][v]


        try:

            distance += float(
                data.get(
                    "length",
                    1.0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            distance += 1.0


        try:

            total_risk += float(
                data.get(
                    "safety_risk",
                    50.0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            total_risk += 50.0


        try:

            score = float(
                data.get(
                    "predicted_safety_score",
                    data.get(
                        "safety_score",
                        50.0
                    )
                )
            )

        except (
            ValueError,
            TypeError
        ):

            score = 50.0


        scores.append(
            score
        )


    average_safety = (

        sum(scores)
        / len(scores)

        if scores

        else 0.0
    )


    return (
        distance,
        total_risk,
        average_safety
    )


# ============================================================
# FASTEST ROUTE
# ============================================================

try:

    fastest_route = nx.shortest_path(
        U,
        start_node,
        end_node,
        weight="length"
    )

except nx.NetworkXNoPath:

    st.error(
        "No route exists between the selected "
        "locations."
    )

    st.stop()

except nx.NodeNotFound:

    st.error(
        "Selected location is not available in "
        "the routing graph."
    )

    st.stop()


fast_distance, fast_risk, fast_safety = route_stats(
    U,
    fastest_route
)


# ============================================================
# ML SAFETY-AWARE ROUTING
# ============================================================

def find_safest_route(
    graph,
    start,
    destination,
    fastest_distance,
    max_detour
):

    maximum_distance = (
        fastest_distance
        *
        (
            1.0
            +
            max_detour / 100.0
        )
    )


    best_route = None

    best_key = None


    try:

        candidates = nx.shortest_simple_paths(
            graph,
            start,
            destination,
            weight="safety_risk"
        )


        for index, route in enumerate(
            candidates
        ):

            if index >= MAX_CANDIDATES:

                break


            distance, risk, score = route_stats(
                graph,
                route
            )


            if distance <= 0:

                continue


            if distance <= maximum_distance:

                # Primary:
                # lowest risk
                #
                # Secondary:
                # shortest distance

                key = (
                    risk,
                    distance
                )


                if (
                    best_key is None
                    or key < best_key
                ):

                    best_key = key

                    best_route = route


    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        pass


    # --------------------------------------------------------
    # No acceptable safe route
    # --------------------------------------------------------

    if best_route is None:

        return (
            fastest_route_fallback(
                graph,
                start,
                destination
            ),
            False
        )


    return (
        best_route,
        True
    )


def fastest_route_fallback(
    graph,
    start,
    destination
):

    return nx.shortest_path(
        graph,
        start,
        destination,
        weight="length"
    )


# ============================================================
# FIND SAFE ROUTE
# ============================================================

safe_route, safe_route_found = (
    find_safest_route(
        U,
        start_node,
        end_node,
        fast_distance,
        max_detour
    )
)


safe_distance, safe_risk, safe_safety = route_stats(
    U,
    safe_route
)


# ============================================================
# COMPARISON
# ============================================================

safety_improvement = (
    safe_safety
    -
    fast_safety
)


risk_reduction = (
    fast_risk
    -
    safe_risk
)


distance_change = (

    (
        safe_distance
        -
        fast_distance
    )
    /
    fast_distance
    *
    100

    if fast_distance > 0

    else 0.0
)


# ============================================================
# RECOMMENDATION
# ============================================================

if (
    sos_active
    and sos_destination
):

    recommendation = "🚨 SOS ROUTE"

    _, emergency_kind, emergency_id = (
        sos_destination
    )

    reason = (
        "Emergency routing is active and the route "
        f"is directed toward the verified "
        f"{emergency_kind} ({emergency_id})."
    )


elif (
    safe_route_found
    and safety_improvement > 0
):

    recommendation = (
        "🛡️ SAFEST ACCEPTABLE ROUTE"
    )

    reason = (
        f"Provides a {safety_improvement:.1f}-point "
        "predicted safety improvement while remaining "
        f"within the {max_detour}% maximum detour."
    )


else:

    recommendation = (
        "⚡ FASTEST ROUTE"
    )

    reason = (
        "No safer alternative within the configured "
        f"{max_detour}% detour limit provides a "
        "meaningful improvement."
    )


# ============================================================
# SOS BANNER
# ============================================================

if (
    sos_active
    and sos_destination
):

    st.error(
        f"🚨 SOS ACTIVE — Routing toward "
        f"{sos_destination[1]} "
        f"({sos_destination[2]})"
    )


elif (
    recommendation
    ==
    "🛡️ SAFEST ACCEPTABLE ROUTE"
):

    st.success(
        "🟢 SafePath found a safer acceptable alternative."
    )


else:

    st.info(
        "🔵 Fastest route is currently the best "
        "acceptable option."
    )


# ============================================================
# MAP COORDINATES
# ============================================================

start_lat = float(
    U.nodes[start_node]["y"]
)

start_lon = float(
    U.nodes[start_node]["x"]
)

end_lat = float(
    U.nodes[end_node]["y"]
)

end_lon = float(
    U.nodes[end_node]["x"]
)


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(
    location=[
        (
            start_lat
            +
            end_lat
        )
        / 2,

        (
            start_lon
            +
            end_lon
        )
        / 2
    ],

    zoom_start=16,

    tiles="CartoDB positron"
)


# ============================================================
# DRAW ML SAFETY ROAD NETWORK
# ============================================================

for u, v, data in U.edges(
    data=True
):

    try:

        u_lat = float(
            U.nodes[u]["y"]
        )

        u_lon = float(
            U.nodes[u]["x"]
        )

        v_lat = float(
            U.nodes[v]["y"]
        )

        v_lon = float(
            U.nodes[v]["x"]
        )


        score = float(
            data.get(
                "predicted_safety_score",
                50.0
            )
        )


        if score >= 70:

            color = "green"

        elif score >= 50:

            color = "orange"

        else:

            color = "red"


        folium.PolyLine(
            [
                [
                    u_lat,
                    u_lon
                ],
                [
                    v_lat,
                    v_lon
                ]
            ],

            color=color,

            weight=3,

            opacity=0.45,

            tooltip=(
                f"ML Safety Score: "
                f"{score:.1f}/100"
            )
        ).add_to(m)


    except (
        KeyError,
        TypeError,
        ValueError
    ):

        continue


# ============================================================
# ROUTE COORDINATES
# ============================================================

def route_coordinates(
    graph,
    route
):

    coordinates = []


    for node in route:

        try:

            coordinates.append(
                [
                    float(
                        graph.nodes[node]["y"]
                    ),
                    float(
                        graph.nodes[node]["x"]
                    )
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue


    return coordinates


fast_coords = route_coordinates(
    U,
    fastest_route
)

safe_coords = route_coordinates(
    U,
    safe_route
)


# ============================================================
# FASTEST ROUTE
# ============================================================

folium.PolyLine(
    fast_coords,

    color="blue",

    weight=5,

    opacity=0.65,

    dash_array="8,8",

    tooltip="⚡ Fastest Route"
).add_to(m)


# ============================================================
# ML SAFETY ROUTE
# ============================================================

folium.PolyLine(
    safe_coords,

    color="green",

    weight=8,

    opacity=0.95,

    tooltip="🛡️ ML Safety-Aware Route"
).add_to(m)


# ============================================================
# START MARKER
# ============================================================

folium.Marker(
    [
        start_lat,
        start_lon
    ],

    tooltip=(
        "📍 "
        +
        node_label(
            start_node
        )
    ),

    popup=(
        "📍 Start<br>"
        +
        node_label(
            start_node
        )
    ),

    icon=folium.Icon(
        color="blue",
        icon="play"
    )
).add_to(m)


# ============================================================
# DESTINATION MARKER
# ============================================================

if (
    sos_active
    and sos_destination
):

    destination_tooltip = (
        f"🚨 {sos_destination[1]}"
        f" — {sos_destination[2]}"
    )

else:

    destination_tooltip = (
        "🏁 "
        +
        node_label(
            end_node
        )
    )


folium.Marker(
    [
        end_lat,
        end_lon
    ],

    tooltip=destination_tooltip,

    popup=destination_tooltip,

    icon=folium.Icon(
        color="red",
        icon="flag"
    )
).add_to(m)


# ============================================================
# MAP / ANALYTICS
# ============================================================

map_column, info_column = st.columns(
    [
        2.25,
        1
    ]
)


# ============================================================
# MAP PANEL
# ============================================================

with map_column:

    st.subheader(
        "🗺️ Live ML Safety-Aware Road Network"
    )


    if (
        sos_active
        and sos_destination
    ):

        st.warning(
            "🚨 Emergency route is active."
        )


    st_folium(
        m,
        width=None,
        height=610,
        returned_objects=[]
    )


# ============================================================
# ANALYTICS PANEL
# ============================================================

with info_column:

    st.markdown(
        "### 📍 Route"
    )


    st.write(
        f"**Start:** "
        f"{node_label(start_node)}"
    )


    if (
        sos_active
        and sos_destination
    ):

        st.write(
            f"**Destination:** "
            f"{sos_destination[1]}"
            f" • "
            f"{sos_destination[2]}"
        )

    else:

        st.write(
            f"**Destination:** "
            f"{node_label(end_node)}"
        )


    st.divider()


    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    a, b = st.columns(2)


    a.metric(
        "Safety",
        f"{safe_safety:.1f}/100"
    )


    b.metric(
        "Distance",
        f"{safe_distance:.0f} m"
    )


    # --------------------------------------------------------
    # Fastest
    # --------------------------------------------------------

    a, b = st.columns(2)


    a.metric(
        "Fastest Safety",
        f"{fast_safety:.1f}"
    )


    b.metric(
        "Detour",
        f"{distance_change:+.1f}%"
    )


    st.divider()


    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Comparison"
    )


    a, b = st.columns(2)


    a.metric(
        "Safety Improvement",
        f"{safety_improvement:+.1f}"
    )


    b.metric(
        "Risk Reduction",
        f"{risk_reduction:+.1f}"
    )


    st.divider()


    # --------------------------------------------------------
    # Why?
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Why this route?"
    )


    if safety_improvement > 0:

        st.write(
            "• Lower combined predicted road risk"
        )

        st.write(
            "• Better ML safety conditions along "
            "the selected road segments"
        )

        st.write(
            f"• **+{safety_improvement:.1f} "
            "safety points** versus fastest route"
        )

    else:

        st.write(
            "• Fastest route already provides the "
            "best available predicted safety within "
            "the selected limit."
        )


    st.divider()


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    st.markdown(
        "### 🛡️ Decision"
    )


    st.write(
        reason
    )


    st.caption(
        f"Maximum detour: {max_detour}%"
    )


    st.caption(
        f"{U.number_of_nodes()} routing nodes • "
        f"{U.number_of_edges()} road segments"
    )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.divider()


with st.expander(
    "🔬 Technical Details"
):

    st.write(
        "SafePath AI uses Random Forest predicted "
        "safety scores attached to road segments. "
        "The routing engine compares the fastest "
        "route against alternative routes using "
        "predicted safety risk."
    )


    prediction_scores = []


    for _, _, data in G.edges(
        data=True
    ):

        try:

            prediction_scores.append(
                float(
                    data.get(
                        "predicted_safety_score",
                        50.0
                    )
                )
            )

        except (
            ValueError,
            TypeError
        ):

            continue


    if prediction_scores:

        st.write(
            "**ML prediction range:** "
            f"{min(prediction_scores):.2f} – "
            f"{max(prediction_scores):.2f}"
        )

    else:

        st.write(
            "**ML prediction range:** Unavailable"
        )


    st.write(
        "**Active ML field:** "
        "`predicted_safety_score`"
    )


    st.write(
        "**Safety routing field:** "
        "`safety_risk`"
    )


    st.write(
        "**Maximum acceptable detour:** "
        f"{max_detour}%"
    )


    st.write(
        "**Maximum candidate routes:** "
        f"{MAX_CANDIDATES}"
    )


    st.write(
        "**ML road network:** "
        f"{G.number_of_nodes()} nodes / "
        f"{G.number_of_edges()} edges"
    )


    st.write(
        "**Routing network:** "
        f"{U.number_of_nodes()} nodes / "
        f"{U.number_of_edges()} edges"
    )


    st.write(
        "**Emergency amenities file:** "
        f"{'Available' if AMENITIES_PATH.exists() else 'Not found'}"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "SafePath AI • Prototype using data-driven ML safety "
    "estimates and spatial routing. Not a guarantee of "
    "real-world safety."
)