#!/usr/bin/env python3
"""Capability Dependency Mapper - Shows which capabilities enable others"""
class DependencyMapper:
    def map_dependencies(self, entries):
        """Build dependency graph: which capabilities led to others"""
        graph = {}
        sorted_entries = sorted(entries, key=lambda e: e.get('timestamp_utc', ''))
        
        for i in range(len(sorted_entries)-1):
            current = sorted_entries[i].get('capability_mode')
            next_cap = sorted_entries[i+1].get('capability_mode')
            
            if current and next_cap:
                if current not in graph:
                    graph[current] = []
                graph[current].append(next_cap)
        
        return graph
    
    def find_foundational_capabilities(self, graph):
        """Find capabilities that enable many others"""
        enabler_counts = {}
        for source, targets in graph.items():
            enabler_counts[source] = len(set(targets))
        
        return sorted(enabler_counts.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    mapper = DependencyMapper()
    print("✓ Dependency mapper ready")
