def neo4j_to_pyg(driver, cypher_query, node_feature_keys=None, edge_feature_keys=None):
    """
    Convert Neo4j subgraph to PyG Data object
    
    Args:
        driver: Neo4j driver instance
        cypher_query: Cypher query to extract subgraph
        node_feature_keys: List of node property keys to use as features
        edge_feature_keys: List of edge property keys to use as features
    
    Returns:
        PyG Data object
    """
    
    def extract_graph_data(tx):
        result = tx.run(cypher_query)
        nodes_dict = {}
        edges_list = []
        
        for record in result:
            for value in record.values():
                if hasattr(value, 'labels'):  # Node
                    nodes_dict[value.id] = {
                        'labels': list(value.labels),
                        'properties': dict(value)
                    }
                elif hasattr(value, 'type'):  # Relationship
                    edges_list.append({
                        'start': value.start_node.id,
                        'end': value.end_node.id,
                        'type': value.type,
                        'properties': dict(value)
                    })
        
        return nodes_dict, edges_list
    
    # Extract data
    with driver.session() as session:
        nodes_dict, edges_list = session.execute_read(extract_graph_data)
    
    # Create mappings
    node_ids = sorted(nodes_dict.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Create edge index
    edge_index = torch.tensor([
        [node_id_to_idx[edge['start']], node_id_to_idx[edge['end']]]
        for edge in edges_list
    ], dtype=torch.long).t().contiguous()
    
    # Create node features
    if node_feature_keys:
        x = torch.tensor([
            [nodes_dict[nid]['properties'].get(key, 0.0) for key in node_feature_keys]
            for nid in node_ids
        ], dtype=torch.float)
    else:
        x = torch.eye(len(node_ids))  # Identity matrix as default features
    
    # Create edge features
    edge_attr = None
    if edge_feature_keys:
        edge_attr = torch.tensor([
            [edge['properties'].get(key, 0.0) for key in edge_feature_keys]
            for edge in edges_list
        ], dtype=torch.float)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_mapping = node_id_to_idx
    
    return data

# Usage
query = "MATCH (n)-[r]-(m) WHERE n.category = 'important' RETURN n, r, m"
pyg_graph = neo4j_to_pyg(driver, query, 
                        node_feature_keys=['feature1', 'feature2'], 
                        edge_feature_keys=['weight'])
