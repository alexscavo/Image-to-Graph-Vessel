import sys
import pandas as pd
import networkx as nx
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import gzip
import nibabel as nib

class PatchGraphGenerator:

    def __init__(self, node_path, edge_path, vvg_path, patch_mode="centerline"):
        """
        Possible patch modes are: 
            - cutoff (eliminates intersected edges)
            - linear (creates new node on the intersection patch plane and linear connection of the nodes)
            - centerline (creates a new node everytime the centerline intersects the boarders of the plane)
        """
        self.max_node_id = None
        self.node_path = node_path
        self.edge_path = edge_path
        self.vvg_path = vvg_path
        self.patch_mode = patch_mode
        self.G = self.create_graph()
        self.centerline_df = self.vvg_to_df()
        self.G_patch_last = None

    def create_graph(self):
        nodes = pd.read_csv(self.node_path, sep=";", index_col="id")
        edges = pd.read_csv(self.edge_path, sep=";", index_col="id")
        self.max_node_id = len(nodes)
        G = nx.MultiGraph()

        for idxN, node in nodes.iterrows():
            G.add_node(idxN, pos=(float(node["pos_x"]), float(node["pos_y"]), float(node["pos_z"])))
        # add the edges
        for idxE, edge in edges.iterrows():
            G.add_edge(edge["node1id"], edge["node2id"])
        return G

    def vvg_to_df(self):
        # with gzip.open(self.vvg_path, 'rb') as vvg_file:
        #     data = json.load(vvg_file)
        
        if self.vvg_path.endswith(".gz"):
            f = gzip.open(self.vvg_path, "rb")
        else:
            f = open(self.vvg_path, "r")

        with f as vvg_file:
            data = json.load(vvg_file)

            id_col = []
            pos_col = []
            node1_col = []
            node2_col = []

            # iterating over all edges in the graph
            for i in data["graph"]["edges"]:
                positions = []
                id_col.append(i["id"])
                node1_col.append(i["node1"])
                node2_col.append(i["node2"])

                # iterating over all the centerline points
                try: 
                    for j in i["skeletonVoxels"]:
                        positions.append(np.array(j["pos"]))
                except Exception as e:
                    # print('-'*100)
                    # print(i)
                    # print('-'*100)
                    pass
                    
                pos_col.append(positions)

            d = {'id_col': id_col, 'pos_col': pos_col, "node1_col": node1_col, "node2_col": node2_col}
            
            df = pd.DataFrame(d)
            df.set_index('id_col')
            return df

    def check_position(self, position, patch_size):
        keep = True
        for dim in range(3):
            if position[dim] < patch_size[dim, 0] or position[dim] > patch_size[dim, 1]:
                keep = False
                break
        return keep

    def create_patch_graph(self, patch_size):

        if self.patch_mode not in ["cutoff", "linear", "centerline"]:
            raise ValueError("Unsupported patch_mode! Use one of: " + str(["cutoff", "linear", "centerline"]))

        elif self.patch_mode == "cutoff":
            subG = self.create_patch_graph_cutoff(patch_size)
            self.G_patch_last = (subG, self.patch_mode, patch_size)


        elif self.patch_mode == "linear":
            subG = None

        elif self.patch_mode == "centerline":
            subG = self.create_patch_graph_centerline(patch_size)
            self.G_patch_last = (subG, self.patch_mode, patch_size)

        return subG

    def create_patch_graph_cutoff(self, patch_size):
        keep_nodes = []
        for node in self.G.nodes():
            position = self.G.nodes[node]["pos"]
            keep = self.check_position(position, patch_size)
            if keep:
                keep_nodes.append(node)

        return nx.subgraph(self.G, keep_nodes)

    # def create_patch_graph_centerline(self, patch_size):
        
    #     keep_nodes = []
    #     for node in self.G.nodes():
    #         keep = True
    #         position = self.G.nodes[node]["pos"]
    #         keep = self.check_position(position, patch_size)
    #         if keep:
    #             # print('mantenuto! node:\n', node, '-'*50)
    #             keep_nodes.append(node)
            

    #     candidate_edges_p1 = self.centerline_df[
    #         self.centerline_df["node1_col"].isin(keep_nodes)]  # and self.centerline_df["node2_col"] not in keep_nodes
    #     candidate_edges_p1 = candidate_edges_p1[~candidate_edges_p1["node2_col"].isin(keep_nodes)]

    #     candidate_edges_p2 = self.centerline_df[
    #         ~self.centerline_df["node1_col"].isin(keep_nodes)]  # and self.centerline_df["node2_col"] not in keep_nodes
    #     candidate_edges_p2 = candidate_edges_p2[candidate_edges_p2["node2_col"].isin(keep_nodes)]

    #     new_node = []
    #     con_to = []

    #     for _, row in candidate_edges_p1.iterrows():
    #         prev_status = None
    #         prev_position = row["pos_col"][0]
    #         for cl_pos in row["pos_col"][1:]:
    #             status = self.check_position(cl_pos, patch_size)

    #             if status == False and prev_status == True:
    #                 new_node.append(prev_position)
    #                 con_to.append(row["node1_col"])
    #                 break
    #             prev_status = status
    #             prev_position = cl_pos

    #     for _, row in candidate_edges_p2.iterrows():
    #         prev_status = None
    #         li = row["pos_col"].copy()
    #         li.reverse()
    #         prev_position = li[0]
    #         for cl_pos in li[1:]:
    #             status = self.check_position(cl_pos, patch_size)

    #             if status == False and prev_status == True:
    #                 new_node.append(prev_position)
    #                 con_to.append(row["node2_col"])
    #                 break

    #             prev_status = status
    #             prev_position = cl_pos

    #     subG = nx.subgraph(self.G.copy(), keep_nodes).copy()

    #     for i, node_pos in enumerate(new_node):
    #         subG.add_edge(self.max_node_id, con_to[i])
    #         subG.add_node(self.max_node_id, pos=node_pos)
    #         self.max_node_id += 1

    #     return subG
    
    def create_patch_graph_centerline(self, patch_size):
        """
        Build a patch graph where truncated edges are kept by inserting
        new nodes on the patch boundary, using the centerline samples.

        Cases:
        1) exactly one endpoint of the edge is inside the patch:
            - add a node at the last 'inside' centerline point
                and connect it to the inside endpoint.
        2) both endpoints are outside, but the centerline passes
            through the patch:
            - add two nodes: first and last 'inside' centerline
                points, and connect them with an edge.
        """

        # -------------------------
        # 1) original nodes kept if inside
        # -------------------------
        keep_nodes = []
        for node in self.G.nodes():
            position = self.G.nodes[node]["pos"]
            
            if self.check_position(position, patch_size):
                keep_nodes.append(node)

        # edges with exactly one endpoint inside
        candidate_edges_p1 = self.centerline_df[
            self.centerline_df["node1_col"].isin(keep_nodes)
        ]
        candidate_edges_p1 = candidate_edges_p1[
            ~candidate_edges_p1["node2_col"].isin(keep_nodes)
        ]

        candidate_edges_p2 = self.centerline_df[
            ~self.centerline_df["node1_col"].isin(keep_nodes)
        ]
        candidate_edges_p2 = candidate_edges_p2[
            candidate_edges_p2["node2_col"].isin(keep_nodes)
        ]
    

        # edges with **both** endpoints outside, but whose centerline may
        # pass through the patch
        candidate_edges_both_out = self.centerline_df[
            ~self.centerline_df["node1_col"].isin(keep_nodes)
        ]
        candidate_edges_both_out = candidate_edges_both_out[
            ~candidate_edges_both_out["node2_col"].isin(keep_nodes)
        ]       

        new_node_pos = []   # positions for new nodes (one-inside edges)
        connect_to = []     # which existing node to connect them to
        segment_pairs = []  # for both-outside edges: (start_pos, end_pos)

        # -------------------------
        # 2) one-inside / one-outside edges
        # -------------------------
        # node1 inside, node2 outside
        for _, row in candidate_edges_p1.iterrows():
            cl_list = list(row["pos_col"])
            if len(cl_list) < 2:
                continue

            prev_position = cl_list[0]
            prev_status = self.check_position(prev_position, patch_size)

            for cl_pos in cl_list[1:]:
                status = self.check_position(cl_pos, patch_size)

                # we just left the patch: last inside point is prev_position
                if (status is False) and (prev_status is True):
                    new_node_pos.append(prev_position)
                    connect_to.append(row["node1_col"])
                    break

                prev_status = status
                prev_position = cl_pos

        # node2 inside, node1 outside → just reverse the centerline
        for _, row in candidate_edges_p2.iterrows():
            cl_list = list(row["pos_col"])[::-1]  # reversed
            if len(cl_list) < 2:
                continue

            prev_position = cl_list[0]
            prev_status = self.check_position(prev_position, patch_size)

            for cl_pos in cl_list[1:]:
                status = self.check_position(cl_pos, patch_size)

                if (status is False) and (prev_status is True):
                    new_node_pos.append(prev_position)
                    connect_to.append(row["node2_col"])
                    break

                prev_status = status
                prev_position = cl_pos

        # -------------------------
        # 3) both-outside edges: build segments fully inside the patch
        # -------------------------
        for _, row in candidate_edges_both_out.iterrows():
            cl_list = list(row["pos_col"])
            if len(cl_list) == 0:
                continue

            prev_pos = cl_list[0]
            prev_status = self.check_position(prev_pos, patch_size)

            inside_run = prev_status
            run_start = prev_pos if inside_run else None

            for cl_pos in cl_list[1:]:
                status = self.check_position(cl_pos, patch_size)

                # entering the patch
                if status and not prev_status:
                    inside_run = True
                    run_start = cl_pos

                # leaving the patch
                if (not status) and prev_status and inside_run:
                    inside_run = False
                    run_end = prev_pos
                    segment_pairs.append((run_start, run_end))
                    run_start = None

                prev_pos = cl_pos
                prev_status = status

            # if we ended still inside patch, close the last segment
            if inside_run and run_start is not None:
                run_end = prev_pos
                segment_pairs.append((run_start, run_end))

        # -------------------------
        # 4) build subgraph and add new nodes/edges
        # -------------------------
        subG = nx.subgraph(self.G.copy(), keep_nodes).copy()

        # one-inside / one-outside: add node + edge to existing node
        for i, node_pos in enumerate(new_node_pos):
            subG.add_node(self.max_node_id, pos=node_pos)
            subG.add_edge(self.max_node_id, connect_to[i])
            self.max_node_id += 1

        # both-outside: for each inside segment, add two nodes and one edge
        for start_pos, end_pos in segment_pairs:
            n_start = self.max_node_id
            subG.add_node(n_start, pos=start_pos)
            self.max_node_id += 1

            n_end = self.max_node_id
            subG.add_node(n_end, pos=end_pos)
            self.max_node_id += 1

            subG.add_edge(n_start, n_end)

        self.G_patch_last = (subG, self.patch_mode, patch_size)
        
        return subG



    def create_patch_graph_linear(self, patch_size):
        pass

    def show_last_patch(self):
        if self.G_patch_last is None:
            raise AttributeError("There is no generated patch graph.")
        G = self.G_patch_last[0]

        # Extract node and edge positions from the layout
        node_xyz = np.array([G.nodes[node]["pos"] for node in G.nodes()])
        edge_xyz = np.array([(G.nodes[u]["pos"], G.nodes[v]["pos"]) for u, v in G.edges()])
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        fig.tight_layout()
        plt.show()

    def show_last_patch_mesh(self, mask):
        if self.G_patch_last is None:
            raise AttributeError("There is no generated patch graph.")

        mask_nii = nib.load(mask)
        mask = np.array(mask_nii.dataobj)
        mask = np.reshape(mask, mask.shape[:3], "C")
        p_dim = self.G_patch_last[2]
        x_l, x_h = p_dim[0, 0], p_dim[0, 1]
        y_l, y_h = p_dim[1, 0], p_dim[1, 1]
        z_l, z_h = p_dim[2, 0], p_dim[2, 1]
        mask = mask[x_l:x_h, y_l:y_h, z_l:z_h]

        G = self.G_patch_last[0]

        # Extract node and edge positions from the layout
        node_xyz = np.array([G.nodes[node]["pos"] for node in G.nodes()])
        node_xyz[:, 0] = node_xyz[:, 0] - x_l
        node_xyz[:, 1] = node_xyz[:, 1] - y_l
        node_xyz[:, 2] = node_xyz[:, 2] - z_l

        edge_xyz = np.array([(G.nodes[u]["pos"], G.nodes[v]["pos"]) for u, v in G.edges()])
        edge_xyz[:, :, 0] = edge_xyz[:, :, 0] - x_l
        edge_xyz[:, :, 1] = edge_xyz[:, :, 1] - y_l
        edge_xyz[:, :, 2] = edge_xyz[:, :, 2] - z_l

        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")
        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        verts, faces, normals, values = measure.marching_cubes(mask, 0)

        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.set_xlim(0, x_h - x_l)
        ax.set_ylim(0, y_h - y_l)
        ax.set_zlim(0, z_h - z_l)

        fig.tight_layout()
        plt.show()

    def get_last_patch(self):

        if self.G_patch_last is None:
            raise AttributeError("There is no generated patch graph.")

        node_names = self.G_patch_last[0].nodes
        rename_dict = dict(zip(node_names, np.arange(len(node_names))))        
        G_sub_relab = nx.relabel_nodes(self.G_patch_last[0], rename_dict)

        node_list = [G_sub_relab.nodes[n]["pos"] for n in G_sub_relab.nodes()]

        edge_list = [e for e in G_sub_relab.edges()]

        return node_list, edge_list

