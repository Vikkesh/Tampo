import os
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import pydotplus

class DAGTask:
    """Represents a single task in a DAG"""
    def __init__(self, task_id: int, data_size: float, cycles: float):
        self.task_id = task_id
        self.data_size = data_size  # bits
        self.cycles = cycles  # CPU cycles required
        self.depth = 0
        self.predecessors = []
        self.successors = []

class DAGParser:
    """Parse DAG files from the data folder"""
    
    def __init__(self, data_folder: str = "data/meta_offloading_20/offload_random20_1"):
        self.data_folder = data_folder
    
    @staticmethod
    def safe_float_convert(value):
        """Safely convert value to float, handling quoted strings"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove quotes and convert
            cleaned = value.strip('"').strip("'")
            return float(cleaned)
        return float(value)
    
    def parse_gv_file(self, filepath: str) -> Dict:
        """
        Parse a .gv (Graphviz) file and return DAG structure
        
        Args:
            filepath: Path to .gv file
            
        Returns:
            Dictionary containing DAG information
        """
        try:
            # Read using pydotplus
            dot_graph = pydotplus.graphviz.graph_from_dot_file(filepath)
            
            tasks = []
            edges = []
            
            # Parse nodes (tasks)
            for node in dot_graph.get_node_list():
                node_name = node.get_name().strip('"')
                if node_name in ['graph', 'node', 'edge']:
                    continue
                    
                task_id = int(node_name)
                
                # Get attributes
                attrs = node.obj_dict.get('attributes', {})
                
                # Safely convert attributes to float
                data_size = self.safe_float_convert(attrs.get('size', 1e6))
                cycles = self.safe_float_convert(attrs.get('expect_size', 1e9))
                
                tasks.append({
                    'id': task_id,
                    'data_size': data_size,
                    'cycles': cycles
                })
            
            tasks.sort(key=lambda task: task['id'])
            id_to_index = {task['id']: idx for idx, task in enumerate(tasks)}
            
            # Parse edges (dependencies)
            for edge in dot_graph.get_edge_list():
                src = int(edge.get_source().strip('"'))
                dst = int(edge.get_destination().strip('"'))
                
                attrs = edge.obj_dict.get('attributes', {})
                comm_size = self.safe_float_convert(attrs.get('size', 0))
                
                edges.append({
                    'source': id_to_index[src],
                    'target': id_to_index[dst],
                    'data': comm_size
                })
            
            adj_matrix = np.zeros((len(tasks), len(tasks)))
            for edge in edges:
                adj_matrix[edge['source'], edge['target']] = 1

            return {
                'num_tasks': len(tasks),
                'tasks': tasks,
                'edges': edges,
                'adj_matrix': adj_matrix
            }
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def load_dataset(self, num_graphs: int = 10, offset: int = 0) -> List[Dict]:
        """
        Load multiple DAG graphs from the data folder
        
        Args:
            num_graphs: Number of graphs to load
            offset: Number of files to skip (useful for train/test splits)
            
        Returns:
            List of DAG dictionaries
        """
        graphs = []
        
        # Get all .gv files in the folder
        gv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.gv')]
        gv_files.sort()
        
        target_files = gv_files[offset:offset + num_graphs]
        for i, filename in enumerate(target_files):
            filepath = os.path.join(self.data_folder, filename)
            dag = self.parse_gv_file(filepath)
            
            if dag is not None:
                graphs.append(dag)
                
        print(f"Loaded {len(graphs)} DAG graphs from {self.data_folder} (offset={offset})")
        return graphs
    
    def calculate_task_priorities(self, dag: Dict) -> np.ndarray:
        """
        Calculate upward rank priorities for HEFT algorithm
        
        Args:
            dag: DAG dictionary
            
        Returns:
            Array of priorities for each task
        """
        num_tasks = dag['num_tasks']
        priorities = np.zeros(num_tasks)
        
        # Build adjacency structure
        successors = [[] for _ in range(num_tasks)]
        for edge in dag['edges']:
            successors[edge['source']].append(edge['target'])
        
        # Calculate priorities (upward rank)
        def calc_rank(task_id: int) -> float:
            if priorities[task_id] > 0:
                return priorities[task_id]
            
            task = dag['tasks'][task_id]
            avg_comp_cost = task['cycles'] / 5e9  # Average computation cost
            
            if len(successors[task_id]) == 0:
                # Exit task
                rank_val = avg_comp_cost
            else:
                # Max of successor ranks + communication cost
                max_succ = 0
                for succ_id in successors[task_id]:
                    succ_rank = calc_rank(succ_id)
                    
                    # Find edge data
                    comm_cost = 0
                    for edge in dag['edges']:
                        if edge['source'] == task_id and edge['target'] == succ_id:
                            comm_cost = edge['data'] / 20e6  # Average bandwidth
                            break
                    
                    max_succ = max(max_succ, succ_rank + comm_cost)
                
                rank_val = avg_comp_cost + max_succ
            
            priorities[task_id] = rank_val
            return rank_val
        
        # Calculate for all tasks
        for i in range(num_tasks):
            calc_rank(i)
        
        return priorities
