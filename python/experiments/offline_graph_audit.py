"""Inspect cached OSM GraphML without an OSMnx install or network access.

This checks XML and cached delivery node membership, NOT OSMnx loading/snap accuracy.
"""
from pathlib import Path
from collections import Counter
import json
import pandas as pd
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / 'maps/xedu_drive.graphml'
SNAPPED = ROOT / 'results/xedu_delivery_road_nodes_scc.csv'
OUT = ROOT / 'audit_outputs'


def audit_graph(graph_path=GRAPH, snapped_path=SNAPPED, out_dir=OUT):
    snapped = pd.read_csv(snapped_path, dtype={'delivery_id': str, 'road_node': str})
    if snapped.delivery_id.duplicated().any():
        raise ValueError('Duplicate snapped delivery IDs')
    expected = set(snapped.road_node)
    present = set()
    counts = Counter()
    edge_has_length = 0
    edge_has_travel_time = 0
    keys = {}
    root_ns = '{http://graphml.graphdrawing.org/xmlns}'
    for _, elem in etree.iterparse(str(graph_path), events=('end',), tag=(root_ns+'key', root_ns+'node', root_ns+'edge')):
        tag = elem.tag.split('}')[-1]
        if tag == 'key':
            keys[elem.get('id')] = elem.get('attr.name')
        elif tag == 'node':
            counts['nodes'] += 1
            node_id = elem.get('id')
            if node_id in expected:
                present.add(node_id)
        elif tag == 'edge':
            counts['edges'] += 1
            attributes = {keys.get(child.get('key')) for child in elem if child.tag.endswith('data')}
            edge_has_length += 'length' in attributes
            edge_has_travel_time += 'travel_time' in attributes
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    result = dict(graph_nodes=counts['nodes'], graph_edges=counts['edges'],
        delivery_count=len(snapped), distinct_snapped_nodes=len(expected),
        snapped_nodes_in_graph=len(present), snapped_nodes_missing=len(expected-present),
        edges_with_length=edge_has_length, edges_with_travel_time=edge_has_travel_time,
        limitations=['XML parse and cached ID membership only',
                     'OSMnx nearest_node and real shortest paths not executed'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'offline_graph_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    print(json.dumps(audit_graph(), indent=2))
