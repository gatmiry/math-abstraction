#!/usr/bin/env python3
"""
Script to create a Ray cluster using apple_bolt.raycluster.

Usage:
    python create_ray_cluster.py
"""

from apple_bolt.raycluster import submit, attach
import ray


def create_cluster():
    """Create and submit a Ray cluster configuration."""
    
    cluster_config = {
        "cluster_name": "My Ray Cluster",
        "ray_version": "2.6.3",
        "head_node_type": "gpu_worker",
        "resources": {
            "docker_image": "docker.apple.com/iris/iris:2.11.0",
        },
        "roles": {
            "gpu_worker": {
                "resources": {
                    "task_type": "1gpu",
                    "max_node": 2,
                    "min_node": 2
                }
            }
        }
    }
    
    print("Creating Ray cluster with configuration:")
    print(f"  Cluster name: {cluster_config['cluster_name']}")
    print(f"  Ray version: {cluster_config['ray_version']}")
    print(f"  Head node type: {cluster_config['head_node_type']}")
    print(f"  Docker image: {cluster_config['resources']['docker_image']}")
    print(f"  GPU worker config: {cluster_config['roles']['gpu_worker']['resources']}")
    
    # Submit the cluster
    # IMPORTANT: The task running this script must be submitted with 'is_parent: true' 
    # in its configuration to allow it to submit child tasks (Ray cluster nodes).
    # This cannot be set from within the script - it must be set when the parent task is submitted.
    print("\nSubmitting cluster...")
    print("NOTE: If you get an error about 'is_parent', the task running this script")
    print("      must be submitted with 'is_parent: true' in its configuration.")
    try:
        cluster = submit(cluster_config)
        print(f"Cluster submitted successfully!")
        print(f"Cluster object: {cluster}")
        return cluster
    except Exception as e:
        print(f"Error submitting cluster: {e}")
        import traceback
        traceback.print_exc()
        raise


def attach_to_cluster(cluster_name=None):
    """Attach to an existing Ray cluster."""
    print(f"Attaching to cluster: {cluster_name or 'default'}")
    try:
        cluster = attach(cluster_name)
        print(f"Successfully attached to cluster: {cluster}")
        return cluster
    except Exception as e:
        print(f"Error attaching to cluster: {e}")
        import traceback
        traceback.print_exc()
        raise


def initialize_ray(cluster=None):
    """Initialize Ray connection."""
    if cluster is not None:
        print("Initializing Ray with cluster connection...")
        # If cluster provides connection info, use it
        # This depends on the apple_bolt.raycluster API
        try:
            ray.init(address=cluster.address if hasattr(cluster, 'address') else None)
            print("Ray initialized successfully!")
            print(f"Ray cluster info: {ray.cluster_resources()}")
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Initializing Ray locally...")
        try:
            ray.init()
            print("Ray initialized successfully!")
            print(f"Ray cluster info: {ray.cluster_resources()}")
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "create":
            # Create new cluster
            cluster = create_cluster()
            if cluster:
                print("\nCluster created. You can now attach to it using:")
                print(f"  python create_ray_cluster.py attach {cluster.name if hasattr(cluster, 'name') else 'My Ray Cluster'}")
        
        elif command == "attach":
            # Attach to existing cluster
            cluster_name = sys.argv[2] if len(sys.argv) > 2 else None
            cluster = attach_to_cluster(cluster_name)
            if cluster:
                initialize_ray(cluster)
        
        elif command == "init":
            # Just initialize Ray (local or existing connection)
            initialize_ray()
        
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python create_ray_cluster.py create    # Create a new cluster")
            print("  python create_ray_cluster.py attach    # Attach to existing cluster")
            print("  python create_ray_cluster.py init      # Initialize Ray locally")
    else:
        # Default: create cluster
        print("No command specified. Creating new cluster...")
        cluster = create_cluster()
        if cluster:
            print("\nTo attach to this cluster later, use:")
            print(f"  python create_ray_cluster.py attach {cluster.name if hasattr(cluster, 'name') else 'My Ray Cluster'}")


if __name__ == "__main__":
    main()

