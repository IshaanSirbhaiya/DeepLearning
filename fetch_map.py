"""
Download the NTU campus walking network from OpenStreetMap and save it
as a GraphML file for fully-offline use.

Run once while you have internet:
    python fetch_map.py
"""

import osmnx as ox

# NTU campus bounding box (generous rectangle around the main campus)
NORTH, SOUTH, EAST, WEST = 1.355, 1.335, 103.690, 103.675

print("⏳  Downloading NTU walking network from OpenStreetMap …")
G = ox.graph_from_bbox(bbox=(WEST, SOUTH, EAST, NORTH), network_type="walk")
print(f"✅  Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

ox.save_graphml(G, filepath="campus.graphml")
print("💾  Saved to campus.graphml – you can now go fully offline!")
